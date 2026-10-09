//! RUC layer-source initialization on the Rust data path.
use super::{clear_error, guard, set_error, ERR, OK};

/// WRF 4.6.1 module_initialize_real.F, account_for_zero_soil_moisture,
/// RUCLSMSCHEME layer arm. Unchanged values keep their exact bits;
/// non-finite inputs remain visible to the cold-start preflight.
pub fn floor_ruc_layer_moisture(values: &mut [f32]) {
    for value in values {
        if value.is_finite() && *value < 0.005_f32 {
            *value = 0.005_f32;
        }
    }
}

/// # Safety
/// `values` must point to `len` writable, aligned, non-aliased f32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_ruc_layer_moisture_floor(
    values: *mut f32, len: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if values.is_null() || len > isize::MAX as usize / std::mem::size_of::<f32>() {
            return set_error("RUC soil moisture pointer or length is invalid");
        }
        floor_ruc_layer_moisture(unsafe { std::slice::from_raw_parts_mut(values, len) });
        OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floor_preserves_unaffected_and_missing_bits() {
        let missing = f32::from_bits(0x7fc01234);
        let mut values = [-0.0214, 0., 0.0049, 0.005, 0.13, 1., missing, f32::NEG_INFINITY, f32::INFINITY];
        let before = values.map(f32::to_bits);
        floor_ruc_layer_moisture(&mut values);
        assert_eq!(&values[..3], &[0.005; 3]);
        assert_eq!(values[3..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before[3..]);
    }
}
