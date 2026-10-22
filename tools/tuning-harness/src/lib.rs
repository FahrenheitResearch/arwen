//! Numerical tools for opt-in product tuning. No engine defaults are changed.

pub mod eki;
pub mod import;
pub mod sample;
pub mod score;

pub type Result<T> = std::result::Result<T, String>;

pub(crate) fn finite(value: f64, context: &str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{context} must be finite"))
    }
}
