//! Array-only smoke preprocessing for the fire process's daily_mean_dcycle
//! arm (gpuwm/chem_fire_reduce.py). No file decoder or remapper is here.
//! Reference: https://raw.githubusercontent.com/ufs-community/ufs-srweather-app/2ad2bc5731819cdc5aef95372a8b27ffe417bf40/ush/smoke_dust/core/cycle.py
//! Lines 252-307 and 330-376. The mean divides by emitting hours, with a
//! two-hour denominator for a single emitting hour, not by all 24 hours.
//! HWP is a finite-value mean of prior physics frames; precipitation sums.
//! Inputs are already remapped kg/cell/hour, MW, dimensionless HWP, and metres.
use std::slice;

/// Arrays use (hour, cell); output is (5, cell): mass, FRP, age, HWP, rain.
/// The caller proves each pointer's length and never aliases output with input.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_fire_daily(
    nc: usize, nh: usize, np: usize, mass: *const f32, frp: *const f32,
    ages: *const f32, hwp: *const f32, rain: *const f32, output: *mut f32,
) -> i32 {
    if nc == 0 || nh != 24 || np == 0 || mass.is_null() || frp.is_null()
        || ages.is_null() || hwp.is_null() || rain.is_null() || output.is_null() {
        return 1;
    }
    let mass = slice::from_raw_parts(mass, nc * nh);
    let frp = slice::from_raw_parts(frp, nc * nh);
    let ages = slice::from_raw_parts(ages, nh);
    let hwp = slice::from_raw_parts(hwp, nc * np);
    let rain = slice::from_raw_parts(rain, nc * np);
    let output = slice::from_raw_parts_mut(output, 5 * nc);
    for c in 0..nc {
        let (mut m, mut f, mut count, mut age) = (0.0_f64, 0.0_f64, 0_usize, 0.0_f32);
        for h in 0..nh {
            let x = h * nc + c;
            if frp[x] > 0.0 {
                if !mass[x].is_finite() || !frp[x].is_finite() || mass[x] < 0.0 {
                    return 2;
                }
                m += mass[x] as f64;
                f += frp[x] as f64;
                if mass[x] != 0.0 { count += 1; }
                age = ages[h]; // Hour order is oldest first: last fire wins.
            }
        }
        let denom = if count < 2 { 2.0 } else { count as f64 };
        if count == 0 { m = 0.0; f = 0.0; age = 0.0; }
        let (mut hw, mut hn, mut pr) = (0.0_f64, 0_usize, 0.0_f64);
        for h in 0..np {
            let x = h * nc + c;
            if hwp[x].is_finite() { hw += hwp[x] as f64; hn += 1; }
            if rain[x].is_finite() && rain[x] > 0.0 { pr += rain[x] as f64; }
        }
        output[c] = (m / denom) as f32;
        output[nc + c] = (f / denom) as f32;
        output[2 * nc + c] = age.max(0.0);
        let mask = m > 0.0 && f > 0.0;
        output[3 * nc + c] = if mask && hn > 0 { (hw / hn as f64) as f32 } else { 0.0 };
        output[4 * nc + c] = if mask { pr as f32 } else { 0.0 };
    }
    0
}
