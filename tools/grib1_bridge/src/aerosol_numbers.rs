//! Table-driven NOAA GSL get_niwfa arithmetic for host preparation.
//! Same provenance and operation order as core/kernels/chem_mp_coupling.cu:
//! ccpp-physics 3e6660c6df54e95a0871e990c2294dd397ae3860,
//! physics/MP/Thompson/mp_thompson.F90:1025-1068 (Apache-2.0).
use crate::{ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};

/// # Safety
/// Each input pointer addresses n FP32 cells; all outputs are disjoint and
/// address n cells. Optional double outputs may be null. Metadata has nr terms.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_aerosol_numbers_f32(
    fields: *const usize, conversion: *const f64, denominator: *const f64,
    target: *const i32, group: *const i32, factor: *const f64,
    nr: usize, n: usize, scale: f64, nifa: *mut f32, nwfa: *mut f32,
    nifa64: *mut f64, nwfa64: *mut f64, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if [fields as usize, conversion as usize, denominator as usize,
            target as usize, group as usize, factor as usize, nifa as usize,
            nwfa as usize].contains(&0) { return ERR_NULL; }
        if nr == 0 || nr > isize::MAX as usize / 8 || n == 0 || n > isize::MAX as usize / 8 {
            return ERR_DIMENSION;
        }
        let f = std::slice::from_raw_parts(fields, nr);
        if f.contains(&0) { return ERR_NULL; }
        let cv = std::slice::from_raw_parts(conversion, nr);
        let den = std::slice::from_raw_parts(denominator, nr);
        let tgt = std::slice::from_raw_parts(target, nr);
        let grp = std::slice::from_raw_parts(group, nr);
        let fac = std::slice::from_raw_parts(factor, nr);
        if !scale.is_finite() || scale <= 0.0 || (0..nr).any(|r|
            !(0..=1).contains(&tgt[r]) || grp[r] < 0 || !cv[r].is_finite() || cv[r] <= 0.0
            || !den[r].is_finite() || den[r] <= 0.0 || !fac[r].is_finite() || fac[r] <= 0.0) {
            return ERR_NONFINITE;
        }
        if (1..nr).any(|r| (tgt[r],grp[r]) < (tgt[r-1],grp[r-1])
            || ((tgt[r],grp[r]) == (tgt[r-1],grp[r-1]) && fac[r] != fac[r-1])) {
            return ERR_DIMENSION;
        }
        let out = [nifa as usize, nwfa as usize];
        let out64 = [nifa64 as usize, nwfa64 as usize];
        let bad = AtomicBool::new(false);
        crate::parallel::run_ranges(n, workers, |start, stop| {
            for cell in start..stop {
                for t in 0..2 {
                    let mut sum = 0.0_f64;
                    let mut subtotal = 0.0_f64;
                    let mut last_group = -1;
                    let mut group_factor = 1.0_f64;
                    for r in 0..nr {
                        if tgt[r] != t as i32 { continue; }
                        if grp[r] != last_group {
                            if last_group >= 0 { sum += subtotal * group_factor; }
                            subtotal = 0.0;
                            last_group = grp[r];
                            group_factor = fac[r];
                        }
                        let value = *((f[r] as *const f32).add(cell));
                        if !value.is_finite() || value < 0.0 { bad.store(true, Ordering::Relaxed); }
                        let mass = f64::from(value) * cv[r];
                        subtotal += mass / den[r];
                    }
                    if last_group >= 0 {
                        sum += subtotal * group_factor;
                        let value = sum * scale;
                        if !(value as f32).is_finite() { bad.store(true, Ordering::Relaxed); }
                        *((out[t] as *mut f32).add(cell)) = value as f32;
                        if out64[t] != 0 { *((out64[t] as *mut f64).add(cell)) = value; }
                    }
                }
            }
        });
        if bad.load(Ordering::Relaxed) { ERR_NONFINITE } else { OK }
    })).unwrap_or(ERR_PANIC)
}
