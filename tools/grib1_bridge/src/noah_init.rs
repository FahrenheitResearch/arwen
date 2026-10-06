//! Noah LSMINIT liquid-water partition with the host reference's f64 order.
//! Columns are independent. The first numerical refusal follows the original
//! column-then-level traversal, regardless of the number of workers.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::portable_math::host_pow;
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

const ERR_CATEGORY: i32 = 40;
const ERR_CATEGORY_RANGE: i32 = 41;
const ERR_LOG_DOMAIN: i32 = 42;
const ERR_ZERO_DIVISION: i32 = 43;
const ERR_POWER_OVERFLOW: i32 = 44;
const ERR_COMPLEX_POWER: i32 = 45;

extern "C" {
    #[link_name = "log"]
    fn host_log(value: f64) -> f64;
    #[link_name = "powf"]
    fn host_powf(base: f32, exponent: f32) -> f32;
}

#[derive(Clone, Copy)]
struct Value { value: f64, kind: u32 }

fn p(value: f64) -> Value { Value { value, kind: 0 } }

impl Value {
    fn promoted(self, other: Self) -> (f64, f64, u32) {
        let kind = self.kind.max(other.kind);
        if kind == 1 { (self.value as f32 as f64, other.value as f32 as f64, kind) }
        else { (self.value, other.value, kind) }
    }
    fn result(value: f64, kind: u32) -> Self {
        Self { value: if kind == 1 { value as f32 as f64 } else { value }, kind }
    }
    fn div(self, other: Self) -> Result<Self, i32> {
        let (left, right, kind) = self.promoted(other);
        if kind == 0 && right == 0.0 { return Err(ERR_ZERO_DIVISION); }
        let value = if kind == 1 { ((left as f32) / (right as f32)) as f64 }
            else { left / right };
        Ok(Self::result(value, kind))
    }
    fn pow(self, other: Self) -> Result<Self, i32> {
        let (base, exponent, kind) = self.promoted(other);
        if kind == 0 && base == 0.0 && exponent < 0.0 { return Err(ERR_ZERO_DIVISION); }
        if kind == 0 && base < 0.0 && base.is_finite() && exponent.is_finite()
            && exponent != exponent.trunc() { return Err(ERR_COMPLEX_POWER); }
        let value = if kind == 1 {
            unsafe { host_powf(base as f32, std::hint::black_box(exponent as f32)) as f64 }
        } else { host_pow(base, exponent) };
        if kind == 0 && value.is_infinite() && base.is_finite() && exponent.is_finite() {
            return Err(ERR_POWER_OVERFLOW);
        }
        Ok(Self::result(value, kind))
    }
    fn gt(self, other: Self) -> bool { let (a, b, _) = self.promoted(other); a > b }
    fn lt(self, other: Self) -> bool { let (a, b, _) = self.promoted(other); a < b }
    fn le(self, other: Self) -> bool { let (a, b, _) = self.promoted(other); a <= b }
    fn abs(self) -> Self { Self { value: self.value.abs(), kind: self.kind } }
    fn log(self) -> Result<Self, i32> {
        if self.value <= 0.0 { Err(ERR_LOG_DOMAIN) }
        else { Ok(p(unsafe { host_log(self.value) })) }
    }
}

impl std::ops::Add for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let (left, right, kind) = self.promoted(other);
        Self::result(if kind == 1 { ((left as f32) + (right as f32)) as f64 }
            else { left + right }, kind)
    }
}
impl std::ops::Sub for Value {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let (left, right, kind) = self.promoted(other);
        Self::result(if kind == 1 { ((left as f32) - (right as f32)) as f64 }
            else { left - right }, kind)
    }
}
impl std::ops::Mul for Value {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (left, right, kind) = self.promoted(other);
        Self::result(if kind == 1 { ((left as f32) * (right as f32)) as f64 }
            else { left * right }, kind)
    }
}
impl std::ops::Neg for Value {
    type Output = Self;
    fn neg(self) -> Self { Self { value: -self.value, kind: self.kind } }
}

fn first_minimum(first: f64, second: f64) -> f64 {
    if second < first { second } else { first }
}

/// Check the integer/finiteness contract before a long-double transport cast
/// can round away its fractional bits. Format 0 is x87 extended in a padded
/// 16-byte word; format 1 is IEEE binary128. Endianness applies to the whole
/// stored word, including NumPy's non-native byte-order dtypes.
///
/// # Safety
/// Source contains length*16 bytes, target contains length f64 elements,
/// and error_index is one writable usize. Out-of-f64-range integer categories
/// saturate for the later category-range refusal without losing their source
/// value in the Python error message.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_noah_category_scan_extended(
    source: *const u8, length: usize, format: u32, big_endian: u32,
    target: *mut f64, error_index: *mut usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if format > 1 || big_endian > 1 || length.checked_mul(16).is_none() { return ERR_DIMENSION; }
        if error_index.is_null() || (length > 0 && (source.is_null() || target.is_null())) { return ERR_NULL; }
        *error_index = usize::MAX;
        for index in 0..length {
            let bytes: [u8; 16] = std::slice::from_raw_parts(source.add(index * 16), 16)
                .try_into().unwrap();
            let word = if big_endian == 1 { u128::from_be_bytes(bytes) }
                else { u128::from_le_bytes(bytes) };
            let (exponent, significand, fraction_bits, normalized) = if format == 0 {
                let exponent = ((word >> 64) & 0x7fff) as i32;
                let significand = word & u64::MAX as u128;
                (exponent, significand, 63, exponent == 0 || significand >> 63 == 1)
            } else {
                let exponent = ((word >> 112) & 0x7fff) as i32;
                let significand = (word & ((1_u128 << 112) - 1))
                    | if exponent == 0 { 0 } else { 1_u128 << 112 };
                (exponent, significand, 112, true)
            };
            let unbiased = exponent - 16383;
            let integer = if exponent == 0x7fff || !normalized { false }
                else if significand == 0 { true }
                else if unbiased < 0 { false }
                else if unbiased >= fraction_bits { true }
                else { significand & ((1_u128 << (fraction_bits - unbiased)) - 1) == 0 };
            if !integer {
                *error_index = index;
                return ERR_CATEGORY;
            }
            let negative = if format == 0 { word & (1_u128 << 79) != 0 }
                else { word & (1_u128 << 127) != 0 };
            let absolute = if significand == 0 { 0.0 }
                else if unbiased >= 1024 { f64::MAX }
                else {
                    let value = (significand as f64) * 2.0_f64.powi(unbiased - fraction_bits);
                    if value.is_infinite() { f64::MAX } else { value }
                };
            *target.add(index) = if negative { -absolute } else { absolute };
        }
        OK
    })).unwrap_or(ERR_PANIC)
}

fn frh2o(temperature: Value, moisture: Value, liquid: Value, maximum: Value,
         exponent: Value, suction: Value) -> Result<(f64, usize, bool), i32> {
    let bx = if exponent.le(p(5.5)) { exponent } else { p(5.5) };
    if temperature.gt(p(273.15) - p(1.0e-3)) {
        return Ok((moisture.value, 0, false));
    }
    let mut frozen = moisture - liquid;
    if frozen.gt(moisture - p(0.02)) { frozen = moisture - p(0.02); }
    if frozen.lt(p(0.0)) { frozen = p(0.0); }
    for iteration in 1..=10 {
        let square = (p(1.0) + p(8.0) * frozen).pow(p(2.0))?;
        let ratio = maximum.div(moisture - frozen)?;
        let first = ((suction * p(9.81)).div(p(3.335e5))? * square * ratio.pow(bx)?).log()?;
        let second = (-(temperature - p(273.15))).div(temperature)?.log()?;
        let difference = first - second;
        let denominator = (p(2.0) * p(8.0)).div(p(1.0) + p(8.0) * frozen)?
            + bx.div(moisture - frozen)?;
        let mut candidate = frozen - difference.div(denominator)?;
        if candidate.gt(moisture - p(0.02)) { candidate = moisture - p(0.02); }
        if candidate.lt(p(0.0)) { candidate = p(0.0); }
        let change = (candidate - frozen).abs();
        frozen = candidate;
        if change.le(p(0.005)) { return Ok(((moisture - frozen).value, iteration, false)); }
    }
    let base = p(3.335e5).div(p(9.81) * (-suction))?
        * (temperature - p(273.15)).div(temperature)?;
    let mut fallback = base.pow(p(-1.0).div(bx)?)? * maximum;
    if fallback.lt(p(0.02)) { fallback = p(0.02); }
    let selected = if moisture.lt(fallback) { moisture } else { fallback };
    Ok((selected.value, 10, true))
}

/// Evaluate one scalar FRH2O call. Each two-bit kind is Python (0), NumPy
/// float32 (1), or NumPy float64 (2), from temperature through suction.
///
/// # Safety
/// Output pointers have space for one value; no pointer overlaps an input.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_noah_frh2o_f64(
    temperature: f64, moisture: f64, liquid: f64, maximum: f64,
    exponent: f64, suction: f64, kinds: u32, output: *mut f64,
    iterations: *mut usize, fallback: *mut u32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if output.is_null() || iterations.is_null() || fallback.is_null() { return ERR_NULL; }
        if kinds >> 12 != 0 { return ERR_DIMENSION; }
        let values = [temperature, moisture, liquid, maximum, exponent, suction];
        let mut typed = [p(0.0); 6];
        for (index, value) in values.into_iter().enumerate() {
            let kind = (kinds >> (2 * index)) & 3;
            if kind == 3 { return ERR_DIMENSION; }
            typed[index] = Value::result(value, kind);
        }
        let [temperature, moisture, liquid, maximum, exponent, suction] = typed;
        match frh2o(temperature, moisture, liquid, maximum, exponent, suction) {
            Ok((value, count, used_fallback)) => {
                *output = value;
                *iterations = count;
                *fallback = u32::from(used_fallback);
                OK
            }
            Err(code) => code,
        }
    })).unwrap_or(ERR_PANIC)
}

/// Initialize a level-major array. Each table row is (bexp, smcmax, psisat).
/// Categories are checked in column order before independent arithmetic.
///
/// # Safety
/// Moisture, temperature and output have levels*columns f64 elements;
/// categories has columns elements, table has soil_rows*3 elements, and
/// error_column has one writable usize. Input/output buffers do not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_noah_initialize_sh2o_f64(
    moisture: *const f64, temperature: *const f64, categories: *const f64,
    table: *const f64, soil_rows: usize, levels: usize, columns: usize,
    workers: usize, output: *mut f64, error_column: *mut usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 || soil_rows == 0 || levels.checked_mul(columns).and_then(|n| n.checked_mul(256)).is_none()
            || soil_rows.checked_mul(3).is_none() { return ERR_DIMENSION; }
        if error_column.is_null() { return ERR_NULL; }
        *error_column = usize::MAX;
        if columns == 0 { return OK; }
        if moisture.is_null() || temperature.is_null() || categories.is_null()
            || table.is_null() || output.is_null() { return ERR_NULL; }
        for column in 0..columns {
            let category = *categories.add(column);
            if !category.is_finite() || category != category.floor() {
                *error_column = column;
                return ERR_CATEGORY;
            }
        }
        let mut valid_columns = columns;
        for column in 0..columns {
            let category = *categories.add(column);
            if category < 1.0 || category > soil_rows as f64 {
                valid_columns = column;
                break;
            }
        }
        if levels == 0 {
            if valid_columns == columns { return OK; }
            *error_column = valid_columns;
            return ERR_CATEGORY_RANGE;
        }
        // A later category refusal must not hide an earlier column's
        // arithmetic failure. Only the valid prefix is evaluated.
        let first_error = AtomicUsize::new(if valid_columns == columns {
            usize::MAX
        } else {
            valid_columns * levels * 256 + ERR_CATEGORY_RANGE as usize
        });
        let pointers = [moisture as usize, temperature as usize, categories as usize,
            table as usize, output as usize];
        crate::parallel::run_ranges(valid_columns, workers, |start, stop| {
            let [moisture, temperature, categories, table, output] = pointers;
            for column in start..stop {
                let row = (*(categories as *const f64).add(column) as usize - 1) * 3;
                let exponent = *(table as *const f64).add(row);
                let maximum = *(table as *const f64).add(row + 1);
                let suction = *(table as *const f64).add(row + 2);
                let usable = exponent > 0.0 && maximum > 0.0 && suction > 0.0;
                let bx = first_minimum(exponent, 5.5);
                for level in 0..levels {
                    let index = level * columns + column;
                    let water = *(moisture as *const f64).add(index);
                    let heat = *(temperature as *const f64).add(index);
                    let target = (output as *mut f64).add(index);
                    *target = water;
                    if !usable || heat >= (273.149_f32 as f64) { continue; }
                    let base = (3.335e5 / (9.81 * (-suction))) * ((heat - 273.15) / heat);
                    let mut first_guess = host_pow(base, -1.0 / bx) * maximum;
                    if first_guess < 0.02 { first_guess = 0.02; }
                    let guess = first_minimum(first_guess, water);
                    let n = |value| Value { value, kind: 2 };
                    match frh2o(n(heat), n(water), n(guess), n(maximum), n(bx), n(suction)) {
                        Ok((value, _, _)) => *target = value,
                        Err(code) => {
                            // One packed index retains reference traversal and
                            // the concrete error code without racing separate fields.
                            let ordinal = column * levels + level;
                            first_error.fetch_min(ordinal * 256 + code as usize, Ordering::Relaxed);
                        }
                    }
                }
            }
        });
        let error = first_error.load(Ordering::Relaxed);
        if error == usize::MAX { OK }
        else {
            *error_column = (error / 256) / levels;
            (error % 256) as i32
        }
    })).unwrap_or(ERR_PANIC)
}
