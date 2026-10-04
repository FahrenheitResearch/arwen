//! Independent cold-start number cells, preserving the host mirror's widths.
//! No field-sized scratch is allocated per worker. Candidate numbers are
//! published by Python only after the reference's whole-species validation.

use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};
use crate::glibc239_math::powf;
use crate::portable_math::host_pow;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Inputs {
    pub mass: *const f32,
    pub number: *const f32,
    pub inverse_density: *const f64,
    pub temperature: *const f32,
    pub theta: *const f64,
    pub pressure: *const f64,
    pub aerosol: *const f32,
    pub landmask: *const f64,
    pub constants: *const f64,
    pub ice_radii: *const f32,
    pub droplet_ratio: *const f32,
    pub cloud_tables: *const f32,
    pub length: usize,
    pub aerosol_length: usize,
    pub landmask_length: usize,
    pub temperature_mode: u32,
    pub species: u32,
    pub workers: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Summary {
    pub invalid_density: u64,
    pub seeded: u64,
    pub repaired: u64,
    pub needs_land: u64,
    pub invalid_aerosol: u64,
    pub invalid_cloud_entry: u64,
    pub invalid_result: u64,
    pub supercooled: u64,
    pub repaired_min: f64,
    pub repaired_max: f64,
}

impl Default for Summary {
    fn default() -> Self {
        Self { invalid_density: 0, seeded: 0, repaired: 0, needs_land: 0,
            invalid_aerosol: 0, invalid_cloud_entry: 0, invalid_result: 0, supercooled: 0,
            repaired_min: f64::INFINITY, repaired_max: f64::NEG_INFINITY }
    }
}

fn maximum(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}
fn minimum(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.min(b) }
}

fn rain_seed(v: f32, t: f32, c: &[f64]) -> f32 {
    let n0 = if t <= 271.15_f32 { 8.0e8_f32 as f64 }
        else if t < 273.15_f32 { (8.0_f32 * powf(10.0, 279.15_f32 - t)) as f64 }
        else { 8.0e6_f32 as f64 };
    let lam = ((n0 * c[0] * 6.0) / v as f64).sqrt().sqrt();
    (((v / c[1] as f32) as f64) * lam * lam * lam / c[0]) as f32
}

fn ice_seed(v: f32, t: f32, c: &[f64], retab: &[f32]) -> f32 {
    let raw = (t - 179.0_f32).trunc();
    // NumPy's float-to-int64 conversion yields INT64_MIN on an invalid
    // conversion, including a large positive finite temperature. Rust's
    // saturating cast would instead select the last table row.
    let converted = if !raw.is_finite() || raw as f64 >= 9223372036854775808.0
        || (raw as f64) < -9223372036854775808.0 { i64::MIN } else { raw as i64 };
    let index = converted.clamp(1, 94) as usize;
    let corr = t - t.trunc();
    let reice = retab[index - 1] * (1.0_f32 - corr) + retab[index] * corr;
    let diameter = (2.0_f32 * reice) * 1.0e-6_f32;
    let lam = (3.0_f32 / diameter) as f64;
    (v as f64 * lam * lam * lam / c[2]) as f32
}

fn cloud_seed(v: f32, aerosol_volume: f32, xland: f32, c: &[f64], ratios: &[f32]) -> f32 {
    if aerosol_volume.is_nan() { return f32::NAN; }
    let (diameter, nu) = if aerosol_volume <= 0.0 {
        if xland - 1.5_f32 > 0.0 { (17.0e-6_f32, 12usize) }
        else { (11.0e-6_f32, 4usize) }
    } else {
        let bounded = maximum(99.0e6_f32, minimum(aerosol_volume, 5.0e10_f32));
        let ratio = (2.5e10_f32 / bounded) as f64;
        let nu = (ratio + 0.5).floor().clamp(2.0, 15.0) as usize;
        let x = maximum(1.0, minimum(bounded * 1.0e-9_f32, 10.0)) - 1.0_f32;
        let diameter = (30.0_f32 - (x * 20.0_f32) / c[3] as f32) * 1.0e-6_f32;
        (diameter, nu)
    };
    let lam = (4.0_f64 + nu as f64) / diameter as f64;
    ((v / ratios[nu - 1]) as f64 * lam * lam * lam / c[0]) as f32
}

fn rain_entry(mass: f32, initial: f32, c: &[f64]) -> f32 {
    let mut number = maximum(c[5] as f32, initial);
    let pref = c[7] as f32 * mass;
    if number <= c[5] as f32 {
        number = (pref as f64 * c[8] / c[9]) as f32;
    }
    let argument = ((c[9] as f32 * 6.0_f32) * number) / mass;
    let lam = powf(argument, c[6] as f32) as f64;
    let diameter = (c[10] / lam) as f32;
    if diameter > c[11] as f32 { number = (pref as f64 * c[12] / c[9]) as f32; }
    if diameter < c[13] as f32 { number = (pref as f64 * c[14] / c[9]) as f32; }
    number
}

fn ice_entry(mass: f32, initial: f32, c: &[f64]) -> f32 {
    let mut number = maximum(c[5] as f32, initial);
    let pref = (c[7] as f32 * mass) / c[15] as f32;
    let raw_small = pref as f64 * c[17];
    let small = if raw_small.is_nan() { f64::NAN } else { raw_small.min(c[16]) };
    if number <= c[5] as f32 { number = small as f32; }
    let argument = ((c[15] as f32 * 6.0_f32) * number) / mass;
    let lam = powf(argument, c[6] as f32) as f64;
    let diameter = (4.0_f64 * (1.0_f64 / lam)) as f32;
    if diameter < c[18] as f32 { number = small as f32; }
    else if diameter > c[19] as f32 { number = (pref as f64 * c[20]) as f32; }
    number
}

fn cloud_entry(mass: f32, initial: f32, c: &[f64], tables: &[f32]) -> f64 {
    let number = maximum(c[23] as f32, minimum(initial, c[21] as f32));
    if number.is_nan() { return f64::NAN; }
    let ratio = (1.0e9_f32 / number) as f64;
    let nu = ((ratio + 0.5).floor() + 2.0).clamp(1.0, 15.0) as usize;
    let ccg1 = tables[nu];
    let ccg2 = tables[16 + nu];
    let ocg1 = tables[32 + nu];
    let ocg2 = tables[48 + nu];
    let cce2 = tables[64 + nu];
    let argument = (((number * c[9] as f32) * ccg2) * ocg1) / mass;
    let mut lam = powf(argument, c[6] as f32) as f64;
    let diameter = (((3.0_f32 + nu as f32) + 1.0_f32) as f64 / lam) as f32;
    let small = diameter < c[24] as f32;
    let large = diameter > c[25] as f32 * 2.0_f32;
    if small { lam = (cce2 / c[24] as f32) as f64; }
    if large && !small { lam = (cce2 / (c[25] as f32 * 2.0_f32)) as f64; }
    let pref = (((ccg1 * ocg2) * mass) / c[9] as f32) as f64;
    let result = pref * host_pow(lam, 3.0);
    if result.is_nan() { f64::NAN } else { result.min(c[22]) }
}

/// Transform one complete species into a private candidate array.
///
/// # Safety
/// Fields have `length` elements, except the documented scalar/broadcast
/// lengths and constant tables (28 f64, 95/15/80 f32). All output arrays
/// have `length` elements and do not alias inputs. `seed_volume` and
/// `seed_mask` may both be null when diagnostics are not requested.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_cold_start_numbers_f32(
    inputs: *const Inputs, output: *mut f32, seed_volume: *mut f32,
    seed_mask: *mut u8, summary: *mut Summary,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if inputs.is_null() || summary.is_null() { return ERR_NULL; }
        let input = *inputs;
        if input.species > 2 || input.workers == 0 || input.temperature_mode > 2 {
            return ERR_DIMENSION;
        }
        if input.length == 0 { *summary = Summary::default(); return OK; }
        if input.mass.is_null() || input.number.is_null() || input.inverse_density.is_null()
            || input.constants.is_null() || input.ice_radii.is_null()
            || input.droplet_ratio.is_null() || input.cloud_tables.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if input.temperature_mode == 1 && input.temperature.is_null()
            || input.temperature_mode == 2 && (input.theta.is_null() || input.pressure.is_null()) {
            return ERR_NULL;
        }
        if input.aerosol_length != 0 && input.aerosol_length != 1 && input.aerosol_length != input.length {
            return ERR_DIMENSION;
        }
        if input.aerosol_length != 0 && input.aerosol.is_null()
            || input.landmask_length != 0 && input.landmask.is_null()
            || seed_volume.is_null() != seed_mask.is_null() { return ERR_NULL; }
        if input.landmask_length != 0 && input.length % input.landmask_length != 0 { return ERR_DIMENSION; }
        let constants = std::slice::from_raw_parts(input.constants, 28);
        let retab = std::slice::from_raw_parts(input.ice_radii, 95);
        let ratios = std::slice::from_raw_parts(input.droplet_ratio, 15);
        let tables = std::slice::from_raw_parts(input.cloud_tables, 80);
        let widths = input.workers.min(crate::parallel::resources::available_workers()).min(input.length).max(1);
        let ranges = crate::worker_ranges(input.length, widths);
        let mut partial = vec![Summary::default(); ranges.len()];
        let partial_address = partial.as_mut_ptr() as usize;
        let input_address = inputs as usize;
        let output_address = output as usize;
        let volume_address = seed_volume as usize;
        let mask_address = seed_mask as usize;
        crate::parallel::run_ranges(ranges.len(), widths, |begin, end| {
            let input = unsafe { &*(input_address as *const Inputs) };
            for slot in begin..end {
                let mut local = Summary::default();
                let (start, stop) = ranges[slot];
                for cell in start..stop {
                    let mass = unsafe { *input.mass.add(cell) };
                    let old = unsafe { *input.number.add(cell) };
                    let alt = unsafe { *input.inverse_density.add(cell) } as f32;
                    let mut result = old;
                    let seeded = mass > 0.0 && old <= 0.0;
                    if mask_address != 0 { unsafe { *(mask_address as *mut u8).add(cell) = u8::from(seeded) }; }
                    if alt.is_nan() || alt == 0.0 { local.invalid_density += 1; }
                    else if seeded {
                        local.seeded += 1;
                        let rho = 1.0_f32 / alt;
                        let volume_mass = mass * rho;
                        let temperature = if input.species == 2 { 0.0 } else { match input.temperature_mode {
                            1 => unsafe { *input.temperature.add(cell) },
                            2 => {
                                let theta = unsafe { *input.theta.add(cell) };
                                let pressure = unsafe { *input.pressure.add(cell) };
                                (theta * libm::pow(pressure / constants[26], constants[27])) as f32
                            }
                            _ => 0.0,
                        }};
                        if input.species == 0 && temperature <= 271.15_f32 { local.supercooled += 1; }
                        let seed = match input.species {
                            0 => rain_seed(volume_mass, temperature, constants),
                            1 => ice_seed(volume_mass, temperature, constants, retab),
                            _ => {
                                let aerosol = if input.aerosol_length == 0 { 0.0 }
                                    else { unsafe { *input.aerosol.add(if input.aerosol_length == 1 { 0 } else { cell }) } };
                                let aerosol_volume = aerosol * rho;
                                if aerosol_volume.is_nan() { local.invalid_aerosol += 1; }
                                if aerosol_volume <= 0.0 && input.landmask_length == 0 { local.needs_land += 1; }
                                let xland = if input.landmask_length == 0 { 1.0 }
                                    else if unsafe { *input.landmask.add(cell % input.landmask_length) } >= 0.5 { 1.0 } else { 2.0 };
                                cloud_seed(volume_mass, aerosol_volume, xland, constants, ratios)
                            }
                        };
                        if volume_address != 0 { unsafe { *(volume_address as *mut f32).add(cell) = seed }; }
                        result = seed / rho;
                        if mass > constants[4] as f32 {
                            local.repaired += 1;
                            let density = 1.0_f64 / alt as f64;
                            let m = (mass as f64 * density) as f32;
                            let n = (result as f64 * density) as f32;
                            if input.species == 2 && n.is_nan() { local.invalid_cloud_entry += 1; }
                            let number_volume = match input.species {
                                0 => rain_entry(m, n, constants) as f64,
                                1 => ice_entry(m, n, constants) as f64,
                                _ => cloud_entry(m, n, constants, tables),
                            };
                            result = (number_volume / density) as f32;
                            local.repaired_min = local.repaired_min.min(result as f64);
                            local.repaired_max = local.repaired_max.max(result as f64);
                        }
                    }
                    if !result.is_finite() || result < 0.0 { local.invalid_result += 1; }
                    unsafe { *(output_address as *mut f32).add(cell) = result; }
                }
                unsafe { *(partial_address as *mut Summary).add(slot) = local; }
            }
        });
        let mut combined = Summary::default();
        for part in partial {
            combined.invalid_density += part.invalid_density;
            combined.seeded += part.seeded;
            combined.repaired += part.repaired;
            combined.needs_land += part.needs_land;
            combined.invalid_aerosol += part.invalid_aerosol;
            combined.invalid_cloud_entry += part.invalid_cloud_entry;
            combined.invalid_result += part.invalid_result;
            combined.supercooled += part.supercooled;
            combined.repaired_min = combined.repaired_min.min(part.repaired_min);
            combined.repaired_max = combined.repaired_max.max(part.repaired_max);
        }
        *summary = combined;
        OK
    })).unwrap_or(ERR_PANIC)
}

/// Publish a validated private FP32 candidate on independent ranges.
///
/// # Safety
/// Source and target address `length` elements and do not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_parallel_copy_f32(
    source: *const f32, target: *mut f32, length: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 { return ERR_DIMENSION; }
        if length == 0 { return OK; }
        if source.is_null() || target.is_null() { return ERR_NULL; }
        let source = source as usize;
        let target = target as usize;
        crate::parallel::run_ranges(length, workers, |start, stop| {
            std::ptr::copy_nonoverlapping((source as *const f32).add(start),
                                         (target as *mut f32).add(start), stop - start);
        });
        OK
    })).unwrap_or(ERR_PANIC)
}
