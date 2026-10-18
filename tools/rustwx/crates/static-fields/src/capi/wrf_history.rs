//! WRF initialized cell geometry for the native-input history door.
use super::{clear_error, guard, set_error, ERR, OK};

/// module_physics_init.F:5658-5659 is the active stock arm. The map-factor
/// area and square-root distance above it are disabled by `#if 0`.
///
/// # Safety
/// Both output arrays must contain `count` writable f32 words. The two
/// map-factor pointers are retained in the ABI but are not read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_wrf_cell_geometry(
    dx: f32, dy: f32, _msftx: *const f32, _msfty: *const f32,
    area: *mut f32, distance: *mut f32, count: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if count == 0 || area.is_null() || distance.is_null()
        {
            return set_error("WRF cell geometry needs two nonempty output arrays");
        }
        if !dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0 {
            return set_error("WRF cell geometry needs positive finite spacing");
        }
        let area = unsafe { std::slice::from_raw_parts_mut(area, count) };
        let distance = unsafe { std::slice::from_raw_parts_mut(distance, count) };
        for i in 0..count {
            area[i] = dx * dy;
            distance[i] = dx;
        }
        OK
    })
}

/// Noah LSMINIT replaces native SNOALB only when RDMAXALB is false
/// (module_sf_noahdrv.F:1902-1904). MAXALB and 0.01 are default REAL.
///
/// # Safety
/// `max_alb` has `categories` words; other arrays have `count` words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_wrf_noah_snow_albedo(
    max_alb: *const f32, categories: usize, ivgtyp: *const i32,
    output: *mut f32, count: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if max_alb.is_null() || ivgtyp.is_null() || output.is_null()
            || categories == 0 || count == 0
        {
            return set_error("native Noah snow albedo needs nonempty table and fields");
        }
        let table = unsafe { std::slice::from_raw_parts(max_alb, categories) };
        let indices = unsafe { std::slice::from_raw_parts(ivgtyp, count) };
        if table.iter().any(|v| !v.is_finite())
            || indices.iter().any(|i| *i < 1 || *i as usize > categories)
        {
            return set_error("native Noah snow albedo has invalid table or vegetation category");
        }
        let output = unsafe { std::slice::from_raw_parts_mut(output, count) };
        for (value, index) in output.iter_mut().zip(indices) {
            *value = table[*index as usize - 1] * 0.01_f32;
        }
        OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unit_nominal_geometry() {
        let mut area = [0.0_f32];
        let mut distance = [0.0_f32];
        let status = unsafe { gpuwm_static_wrf_cell_geometry(3000.0, 3000.0,
            [1.0].as_ptr(), [1.0].as_ptr(), area.as_mut_ptr(), distance.as_mut_ptr(), 1) };
        assert_eq!(status, OK);
        assert_eq!(area[0].to_bits(), 9000000.0_f32.to_bits());
        assert_eq!(distance[0].to_bits(), 3000.0_f32.to_bits());
        let status = unsafe { gpuwm_static_wrf_cell_geometry(3000.0, 3000.0,
            [0.0].as_ptr(), [1.0].as_ptr(), area.as_mut_ptr(), distance.as_mut_ptr(), 1) };
        assert_eq!(status, OK);
        assert_eq!(area[0].to_bits(), 9000000.0_f32.to_bits());
    }
    #[test]
    fn noah_source_real_snow_albedo() {
        let mut output = [0.0_f32];
        let status = unsafe { gpuwm_static_wrf_noah_snow_albedo([66.0].as_ptr(), 1,
            [1].as_ptr(), output.as_mut_ptr(), 1) };
        assert_eq!(status, OK);
        assert_eq!(output[0].to_bits(), 0x3f28f5c2);
    }
}
