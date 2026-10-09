//! METGRID nearest_neighbor+four_pt+average_4pt, then fill_missing=0.
//! Donors are always read from the original plane, including its mask.
//! WPS uses floor/ceiling corners, so an exact masked native grid point
//! exhausts the chain. A fractional target can use its finite corners.
//! Authority: WPS v4.6.0 geogrid/src/interp_module.F:360,598,1011.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub masked: usize,
    pub nearest_neighbor: usize,
    pub four_pt: usize,
    pub average_4pt: usize,
    pub zero: usize,
}

impl Counts {
    pub fn add(&mut self, other: Self) {
        self.masked += other.masked;
        self.nearest_neighbor += other.nearest_neighbor;
        self.four_pt += other.four_pt;
        self.average_4pt += other.average_4pt;
        self.zero += other.zero;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage { NearestNeighbor, FourPt, Average4pt, Zero }

/// Apply the same chain at the Python boundary without discarding the
/// original source bitmap. Counts are nearest, four-point, average, zero
/// for every layer. Finite nearest answers retain their original FP32 bits.
///
/// # Safety
/// Buffers must have the dimensions specified here and must not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_missing_value_chain_f32(
    source: *const f32, valid: *const u8, target_y: *const f64, target_x: *const f64,
    output: *mut f32, counts: *mut u64, nlayer: usize, ny: usize, nx: usize,
    ntarget: usize, workers: usize,
) -> i32 {
    use crate::{ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || valid.is_null() || target_y.is_null() || target_x.is_null()
            || output.is_null() || counts.is_null() { return ERR_NULL; }
        if nlayer == 0 || ny == 0 || nx == 0 || ntarget == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        let Some(plane) = nx.checked_mul(ny) else { return ERR_DIMENSION; };
        let Some(length) = plane.checked_mul(nlayer) else { return ERR_DIMENSION; };
        if nlayer.checked_mul(ntarget).is_none() || nlayer.checked_mul(4).is_none() {
            return ERR_DIMENSION;
        }
        let source = std::slice::from_raw_parts(source, length);
        let valid = std::slice::from_raw_parts(valid, length);
        let yy = std::slice::from_raw_parts(target_y, ntarget);
        let xx = std::slice::from_raw_parts(target_x, ntarget);
        if valid.iter().any(|v| *v > 1) { return ERR_DIMENSION; }
        if source.iter().zip(valid).any(|(v, mask)| *mask != 0 && !v.is_finite())
            || yy.iter().zip(xx).any(|(y, x)| !x.is_finite() || !y.is_finite()
                || *x < 0.0 || *y < 0.0 || *x > (nx - 1) as f64 || *y > (ny - 1) as f64) {
            return ERR_NONFINITE;
        }
        let output_address = output as usize;
        let counts_address = counts as usize;
        crate::parallel::run_ranges(nlayer, workers, |start, stop| {
            for layer in start..stop {
                let base = layer * plane;
                let original: Vec<f64> = source[base..base + plane].iter()
                    .zip(&valid[base..base + plane])
                    .map(|(v, mask)| if *mask != 0 { *v as f64 } else { f64::NAN }).collect();
                let mut tally = [0u64; 4];
                for target in 0..ntarget {
                    let (value, stage) = interpolate(&original, nx, ny, xx[target], yy[target]);
                    let slot = match stage { Stage::NearestNeighbor => 0, Stage::FourPt => 1,
                        Stage::Average4pt => 2, Stage::Zero => 3 };
                    tally[slot] += 1;
                    *((output_address as *mut f32).add(layer * ntarget + target)) = value as f32;
                }
                for (slot, count) in tally.iter().enumerate() {
                    *((counts_address as *mut u64).add(layer * 4 + slot)) = *count;
                }
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Coordinates are zero based. The caller guarantees a rectangular plane
/// and a location inside it. Missing donors are represented by NaN.
pub fn interpolate(values: &[f64], nx: usize, ny: usize, x: f64, y: f64) -> (f64, Stage) {
    assert_eq!(values.len(), nx * ny);
    assert!(x >= 0.0 && y >= 0.0 && x <= (nx - 1) as f64 && y <= (ny - 1) as f64);
    let nearest = values[y.round() as usize * nx + x.round() as usize];
    if nearest.is_finite() { return (nearest, Stage::NearestNeighbor); }
    let x0 = x.floor() as usize;
    let x1 = x.ceil() as usize;
    let y0 = y.floor() as usize;
    let y1 = y.ceil() as usize;
    // WPS arithmetic is REAL. Only reconstructed values take this path;
    // every original finite value retains its existing conversion.
    let corners = [values[y0 * nx + x0], values[y1 * nx + x0],
        values[y0 * nx + x1], values[y1 * nx + x1]];
    if corners.iter().all(|v| v.is_finite()) {
        let [a, b, c, d] = corners.map(|v| v as f32);
        let dx = (x - x0 as f64) as f32;
        let dy = (y - y0 as f64) as f32;
        let value = if x0 == x1 && y0 == y1 { a }
            else if x0 == x1 { a * (1.0 - dy) + b * dy }
            else if y0 == y1 { a * (1.0 - dx) + c * dx }
            else { dy * (b * (1.0 - dx) + d * dx)
                + (1.0 - dy) * (a * (1.0 - dx) + c * dx) };
        return (value as f64, Stage::FourPt);
    }
    // WPS's masked terms multiply its negative missing marker by zero.
    // Start with that negative zero so all-negative-zero donors retain
    // their sign. Adding either zero sign to a nonzero finite donor has
    // the same result, and the donor order and division stay unchanged.
    let mut sum = -0.0f32;
    let mut count = 0;
    for value in corners {
        if value.is_finite() { sum += value as f32; count += 1; }
    }
    if count != 0 { return ((sum / count as f32) as f64, Stage::Average4pt); }
    (0.0, Stage::Zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_target_uses_donors_before_zero() {
        let values = [f64::NAN, 2.0, 4.0, 8.0];
        assert_eq!(interpolate(&values, 2, 2, 0.25, 0.25),
            ((14.0f32 / 3.0) as f64, Stage::Average4pt));
        assert_eq!(interpolate(&values, 2, 2, 0.75, 0.25), (2.0, Stage::NearestNeighbor));
        assert_eq!(interpolate(&values, 2, 2, 0.0, 0.0), (0.0, Stage::Zero));
        assert_eq!(interpolate(&[f64::NAN; 4], 2, 2, 0.25, 0.25), (0.0, Stage::Zero));
    }

    #[test]
    fn boundary_mask_prevents_native_zero_from_becoming_a_donor() {
        let source = [0.0f32, 2.0, 4.0, 8.0];
        let valid = [0u8, 1, 1, 1];
        let yy = [0.25, 0.25, 0.0];
        let xx = [0.25, 0.75, 0.0];
        for workers in [1, 3] {
            let mut out = [f32::NAN; 3];
            let mut counts = [99u64; 4];
            let code = unsafe { gpuwm_missing_value_chain_f32(source.as_ptr(), valid.as_ptr(),
                yy.as_ptr(), xx.as_ptr(), out.as_mut_ptr(), counts.as_mut_ptr(), 1, 2, 2, 3, workers) };
            assert_eq!(code, 0);
            assert_eq!(out, [14.0f32 / 3.0, 2.0, 0.0]);
            assert_eq!(counts, [1, 0, 1, 1]);
        }
    }

    #[test]
    fn average_retains_wps_negative_zero_and_nonzero_order() {
        let values = [f64::NAN, -0.0, -0.0, -0.0];
        for (x, y) in [(0.25, 0.0), (0.0, 0.25), (0.25, 0.25)] {
            let (value, stage) = interpolate(&values, 2, 2, x, y);
            assert_eq!(stage, Stage::Average4pt);
            assert_eq!((value as f32).to_bits(), 0x8000_0000);
        }
        let mixed_zero = [f64::NAN, 0.0, -0.0, -0.0];
        assert_eq!((interpolate(&mixed_zero, 2, 2, 0.25, 0.25).0 as f32).to_bits(), 0);
        // The original corner order cancels the two large donors before
        // adding 1. A reordered average would lose that last donor.
        let cancellation = [f64::NAN, -1.0e20, 1.0e20, 1.0];
        let (value, stage) = interpolate(&cancellation, 2, 2, 0.25, 0.25);
        assert_eq!(stage, Stage::Average4pt);
        assert_eq!((value as f32).to_bits(), 0x3eaa_aaab);
    }
}
