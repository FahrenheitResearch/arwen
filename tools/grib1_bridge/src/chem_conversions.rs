//! Source-level boundary unit conversions, before horizontal remapping.
//! UPP 1296eebb295251d0fe1cc697f47058c62d887977, sorc/ncep_post.fd/
//! MDLFLD.f:2442 uses (1/RD)*(P/T), not virtual or dry-air density.
//! params.F:57 defines RD=287.04. No humidity factor belongs in its inverse.
//! https://github.com/NOAA-EMC/UPP/blob/1296eebb295251d0fe1cc697f47058c62d887977/sorc/ncep_post.fd/MDLFLD.f#L2442

/// No state persists between calls. Each cell uses its own source P and T.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_kg_m3_to_ug_kg_dry_f32(
    mass: *const f32, pressure: *const f32, temperature: *const f32,
    output: *mut f32, length: usize, gas_constant: f32,
) -> i32 {
    if mass.is_null() || pressure.is_null() || temperature.is_null() || output.is_null() {
        return 1;
    }
    if length == 0 { return 2; }
    if !gas_constant.is_finite() || gas_constant <= 0.0 { return 3; }
    let mass = std::slice::from_raw_parts(mass, length);
    let pressure = std::slice::from_raw_parts(pressure, length);
    let temperature = std::slice::from_raw_parts(temperature, length);
    let output = std::slice::from_raw_parts_mut(output, length);
    for i in 0..length {
        if !mass[i].is_finite() || mass[i] < 0.0 || !pressure[i].is_finite()
            || pressure[i] <= 0.0 || !temperature[i].is_finite() || temperature[i] <= 0.0 {
            return 3;
        }
        let density = (1.0_f32 / gas_constant) * (pressure[i] / temperature[i]);
        output[i] = (mass[i] / density) * 1e9_f32;
        if !output[i].is_finite() { return 3; }
    }
    0
}

/// Table-ordered weighted source terms, one contiguous source-major buffer.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_boundary_weighted_sum_f32(
    fields: *const f32, weights: *const f32, output: *mut f32,
    nfields: usize, length: usize,
) -> i32 {
    if fields.is_null() || weights.is_null() || output.is_null() { return 1; }
    let Some(total) = nfields.checked_mul(length) else { return 2; };
    if nfields == 0 || length == 0 { return 2; }
    let fields = std::slice::from_raw_parts(fields, total);
    let weights = std::slice::from_raw_parts(weights, nfields);
    let output = std::slice::from_raw_parts_mut(output, length);
    if fields.iter().chain(weights.iter()).any(|value| !value.is_finite()) { return 3; }
    for cell in 0..length {
        let mut sum = 0.0_f32;
        for term in 0..nfields { sum += fields[term * length + cell] * weights[term]; }
        if !sum.is_finite() || sum < 0.0 { return 3; }
        output[cell] = sum;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_uses_the_source_cell() {
        let mass = [1e-9_f32, 1e-9];
        let p = [100000_f32, 50000.0];
        let t = [300_f32, 300.0];
        let mut out = [0_f32; 2];
        assert_eq!(unsafe { gpuwm_kg_m3_to_ug_kg_dry_f32(mass.as_ptr(), p.as_ptr(), t.as_ptr(), out.as_mut_ptr(), 2, 287.04) }, 0);
        assert_eq!(out[1], 2.0 * out[0]);
    }
}
