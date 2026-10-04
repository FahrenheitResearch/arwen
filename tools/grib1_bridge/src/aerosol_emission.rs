//! Operational WRF surface aerosol source, module_initialize_real.F:4424-4430.
//!
//! NOAA-EMC/HRRR WRFV3.9 source, public domain. The WRF notice is retained
//! in licenses/LICENSE-WRF-public-domain.txt. Operation order is Fortran
//! REAL, including both divisions before multiplication by the cell area.
use std::panic::{catch_unwind, AssertUnwindSafe};

#[no_mangle]
pub unsafe extern "C" fn gpuwm_aerosol_surface_mass_f32(
    number: *const f32,
    lower_geopotential: *const f32,
    upper_geopotential: *const f32,
    inverse_density: *const f32,
    output: *mut f32,
    count: usize,
    gravity: f32,
    dx: f32,
    dy: f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if number.is_null() || lower_geopotential.is_null()
            || upper_geopotential.is_null() || inverse_density.is_null()
            || output.is_null() { return 1; }
        if count == 0 || !gravity.is_finite() || gravity <= 0.0
            || !dx.is_finite() || dx <= 0.0 || !dy.is_finite() || dy <= 0.0 {
            return 2;
        }
        let number = std::slice::from_raw_parts(number, count);
        let lower = std::slice::from_raw_parts(lower_geopotential, count);
        let upper = std::slice::from_raw_parts(upper_geopotential, count);
        let alt = std::slice::from_raw_parts(inverse_density, count);
        let output = std::slice::from_raw_parts_mut(output, count);
        for i in 0..count {
            if !number[i].is_finite() || number[i] < 0.0
                || !lower[i].is_finite() || !upper[i].is_finite()
                || upper[i] <= lower[i] || !alt[i].is_finite() || alt[i] <= 0.0 {
                return 3;
            }
            let z1 = (upper[i] - lower[i]) / gravity;
            let airmass = ((1.0_f32 / alt[i]) * z1 * dx) * dy;
            output[i] = (number[i] * 0.000196_f32) * (airmass * 2.0e-10_f32);
        }
        0
    })).unwrap_or(127)
}
