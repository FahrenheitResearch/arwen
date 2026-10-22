//! Periodic longitude and reflected pole donors for regular global forcing.
//!
//! At a pole the next latitude row is the reflected latitude at longitude
//! +180 degrees. Both geographic horizontal vector components change sign
//! because their east/north basis changes sign under that chart transition.
use rayon::prelude::*;
use super::{clear_error, guard, set_error, ERR, OK};

/// Select donor columns and reflect up to three rows past either pole.
///
/// # Safety
/// `source` and `output` must be non-overlapping buffers holding respectively
/// bands*ny*nx and bands*(ny+south+north)*out_nx doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_global_donors_f64(
    source: *const f64, output: *mut f64,
    bands: usize, ny: usize, nx: usize, period: usize,
    first: i64, out_nx: usize, south: usize, north: usize, vector: i32,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if source.is_null() || output.is_null() || bands == 0 || ny < 2
            || period < 2 || period > nx || out_nx == 0
            || south >= ny || north >= ny || south > 3 || north > 3
            || ((south != 0 || north != 0) && period % 2 != 0)
            || !(vector == 0 || vector == 1) {
            return set_error("invalid global donor geometry (pole reflection needs an even longitude period)");
        }
        let Some(source_len) = bands.checked_mul(ny).and_then(|n| n.checked_mul(nx)) else {
            return set_error("global donor source size overflow");
        };
        let Some(out_ny) = ny.checked_add(south).and_then(|n| n.checked_add(north)) else {
            return set_error("global donor latitude size overflow");
        };
        let Some(output_len) = bands.checked_mul(out_ny).and_then(|n| n.checked_mul(out_nx)) else {
            return set_error("global donor output size overflow");
        };
        let input = unsafe { std::slice::from_raw_parts(source, source_len) };
        let result = unsafe { std::slice::from_raw_parts_mut(output, output_len) };
        result.par_chunks_mut(out_nx).enumerate().for_each(|(row, values)| {
            let band = row / out_ny;
            let j = (row % out_ny) as i64 - south as i64;
            let (source_j, reflected) = if j < 0 {
                ((-j) as usize, true)
            } else if j >= ny as i64 {
                ((2 * (ny as i64 - 1) - j) as usize, true)
            } else { (j as usize, false) };
            let shift = if reflected { (period / 2) as i64 } else { 0 };
            let sign = if reflected && vector == 1 { -1.0 } else { 1.0 };
            for (i, value) in values.iter_mut().enumerate() {
                let source_i = (first + i as i64 + shift).rem_euclid(period as i64) as usize;
                *value = sign * input[(band * ny + source_j) * nx + source_i];
            }
        });
        OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reflected_pole_scalars_and_vectors_keep_the_opposite_meridian() {
        let input: Vec<f64> = (0..3*8).map(|v| v as f64 + 1.0).collect();
        let mut scalar = vec![0.0; 5*14];
        let mut vector = scalar.clone();
        unsafe {
            assert_eq!(gpuwm_static_global_donors_f64(input.as_ptr(), scalar.as_mut_ptr(), 1, 3, 8, 8, -3, 14, 1, 1, 0), OK);
            assert_eq!(gpuwm_static_global_donors_f64(input.as_ptr(), vector.as_mut_ptr(), 1, 3, 8, 8, -3, 14, 1, 1, 1), OK);
        }
        for j in 0..5 {
            for i in 0..14 {
                let reflected = j == 0 || j == 4;
                let sj = if reflected {1} else {j-1};
                let si = (i as i64 - 3 + if reflected {4} else {0}).rem_euclid(8) as usize;
                assert_eq!(scalar[j*14+i], input[sj*8+si]);
                assert_eq!(vector[j*14+i], if reflected {-scalar[j*14+i]} else {scalar[j*14+i]});
            }
        }
    }
}
