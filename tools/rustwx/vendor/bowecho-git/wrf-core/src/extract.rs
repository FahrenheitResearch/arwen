//! Convenience wrappers around WrfFile for common field extraction patterns.
//!
//! Most of these delegate to WrfFile's cached methods; this module exists to
//! provide a flat function-based API that the diagnostic modules can call.

use crate::error::WrfResult;
use crate::file::WrfFile;

/// Full 3-D pressure (Pa). `[nz, ny, nx]`
pub fn full_pressure(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.full_pressure(t).map(|v| v.to_vec())
}

/// Full 3-D potential temperature (K). `[nz, ny, nx]`
pub fn full_theta(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.full_theta(t).map(|v| v.to_vec())
}

/// Full 3-D geopotential (m^2/s^2), destaggered. `[nz, ny, nx]`
pub fn full_geopotential(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.full_geopotential(t).map(|v| v.to_vec())
}

/// 3-D temperature (K). `[nz, ny, nx]`
pub fn temperature(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.temperature(t).map(|v| v.to_vec())
}

/// 3-D height MSL (m). `[nz, ny, nx]`
pub fn height_msl(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.height_msl(t).map(|v| v.to_vec())
}

/// 3-D height AGL (m). `[nz, ny, nx]`
pub fn height_agl(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.height_agl(t).map(|v| v.to_vec())
}

/// 2-D terrain height (m). `[ny, nx]`
pub fn terrain(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.terrain(t).map(|v| v.to_vec())
}

/// Water vapour mixing ratio (kg/kg). `[nz, ny, nx]`
pub fn qvapor(f: &WrfFile, t: usize) -> WrfResult<Vec<f64>> {
    f.qvapor(t).map(|v| v.to_vec())
}
