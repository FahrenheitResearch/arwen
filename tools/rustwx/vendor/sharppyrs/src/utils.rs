//! Small display helpers (formatting ports of `sharppy.sharptab.utils`).

/// Quality control: is this a usable value? (port of `utils.QC`)
#[inline]
pub fn qc(val: f64) -> bool {
    val.is_finite() && val > -9990.0
}

/// Convert meters to feet.
#[inline]
pub fn m2ft(val: f64) -> f64 {
    val * 3.2808399
}

/// Format a value as a rounded integer string, `--` when missing
/// (port of `utils.INT2STR`).
pub fn int2str(val: f64) -> String {
    if !qc(val) {
        return "--".to_string();
    }
    format!("{}", val.round() as i64)
}

/// Format a value with the given precision, `--` when missing
/// (port of `utils.FLOAT2STR`).
pub fn float2str(val: f64, precision: usize) -> String {
    if !qc(val) {
        return "--".to_string();
    }
    format!("{val:.precision$}")
}
