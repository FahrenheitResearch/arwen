//! Deterministic parallel CPU kernels for GPUWM native preprocessing.
//!
//! The GRIB bridges and this library intentionally share one Rust package so
//! a distributable preprocessor can reuse the already-vendored decoder build
//! without introducing a second native toolchain.  The C ABI is deliberately
//! small: Python owns metadata/shape validation and passes contiguous FP32
//! buffers; Rust partitions independent target points or columns into fixed,
//! contiguous ranges.  No reduction crosses a worker boundary, so changing
//! the worker count cannot change an output element's arithmetic.

pub mod dealias;
pub mod quantization;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI32, Ordering};

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// artifact was built from, embedded so the release cut can prove a
/// staged binary matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`), never by
/// executing it.  `build.rs` injects the value; `unknown` marks a build
/// the cut must refuse (outside git, or a dirty tree).  Every
/// executable in this package references this constant from `main` so
/// the linker cannot discard the bytes.
pub static SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The stamp for C callers, NUL-terminated -- and the exported
/// reference that keeps the stamp bytes present in the cdylib the
/// release ships, where no `main` exists to hold them.
#[no_mangle]
pub extern "C" fn gpuwm_preprocess_cpu_source_rev()
-> *const std::os::raw::c_char {
    static C_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    C_STAMP.as_ptr().cast()
}

pub(crate) const OK: i32 = 0;
pub(crate) const ERR_NULL: i32 = 1;
pub(crate) const ERR_DIMENSION: i32 = 2;
pub(crate) const ERR_NONFINITE: i32 = 3;
pub(crate) const ERR_PRESSURE_ORDER: i32 = 4;
pub(crate) const ERR_SURFACE_BRACKET: i32 = 5;
pub(crate) const ERR_TARGET_ABOVE_TOP: i32 = 6;
pub(crate) const ERR_INTERPOLATION_WINDOW: i32 = 7;
pub(crate) const ERR_PANIC: i32 = 127;

#[inline]
pub(crate) fn worker_ranges(length: usize, workers: usize) -> Vec<(usize, usize)> {
    let count = workers.min(length);
    let quotient = length / count;
    let remainder = length % count;
    let mut ranges = Vec::with_capacity(count);
    let mut start = 0usize;
    for index in 0..count {
        let stop = start + quotient + usize::from(index < remainder);
        ranges.push((start, stop));
        start = stop;
    }
    ranges
}

#[inline]
fn nearest_even_nonnegative(value: f32) -> usize {
    let lower = value.floor();
    let fraction = value - lower;
    let lower_index = lower as usize;
    if fraction < 0.5 {
        lower_index
    } else if fraction > 0.5 || lower_index % 2 == 1 {
        lower_index + 1
    } else {
        lower_index
    }
}

#[inline]
fn cuda_ftz_product_nonzero(left: f32, right: f32) -> bool {
    // NVIDIA FP32 arithmetic flushes subnormal operands/results to zero.  The
    // WPS `oned` missing-value branch tests the product rather than the two
    // operands independently, so preserving host subnormals here can select a
    // different interpolation polynomial at legitimate zero-valued stencils.
    // Mirror the CUDA authority explicitly instead of depending on host MXCSR
    // state or compiler flags.
    left.is_normal() && right.is_normal() && {
        let product = left * right;
        product.is_normal() || product.is_infinite()
    }
}

#[inline]
fn wps_oned(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let zero = 0.0f32;
    let half = 0.5f32;
    let one = 1.0f32;
    let regular = (one - x) * (b + x * (half * (c - a) + x * (half * (c + a) - b)))
        + x * (c + (one - x) * (half * (b - d) + (one - x) * (half * (b + d) - c)));
    let mut out = zero;
    if x == zero {
        out = b;
    }
    if x == one {
        out = c;
    }
    let both = cuda_ftz_product_nonzero(b, c);
    if both && a == zero && d == zero {
        out = b * (one - x) + c * x;
    }
    if both && a != zero && d == zero {
        out = b + x * (half * (c - a) + x * (half * (c + a) - b));
    }
    if both && a == zero && d != zero {
        out = c + (one - x) * (half * (b - d) + (one - x) * (half * (b + d) - c));
    }
    if both && a != zero && d != zero {
        out = regular;
    }
    out
}

#[inline]
fn source_at(source: &[f32], lead: usize, y: usize, x: usize, ny: usize, nx: usize) -> f32 {
    source[(lead * ny + y) * nx + x]
}

fn horizontal_point(
    source: &[f32],
    lead: usize,
    y: f32,
    x: f32,
    ny: usize,
    nx: usize,
    method: i32,
) -> Result<f32, i32> {
    if !y.is_finite()
        || !x.is_finite()
        || y < 0.0
        || x < 0.0
        || y > (ny - 1) as f32
        || x > (nx - 1) as f32
    {
        return Err(ERR_NONFINITE);
    }
    if method == 0 {
        let iy = nearest_even_nonnegative(y).min(ny - 1);
        let ix = nearest_even_nonnegative(x).min(nx - 1);
        return Ok(source_at(source, lead, iy, ix, ny, nx));
    }
    if method == 1 {
        let iy = (y.floor() as usize).min(ny - 2);
        let ix = (x.floor() as usize).min(nx - 2);
        let fy = y - iy as f32;
        let fx = x - ix as f32;
        let one = 1.0f32;
        let lower = (one - fx) * source_at(source, lead, iy, ix, ny, nx)
            + fx * source_at(source, lead, iy, ix + 1, ny, nx);
        let upper = (one - fx) * source_at(source, lead, iy + 1, ix, ny, nx)
            + fx * source_at(source, lead, iy + 1, ix + 1, ny, nx);
        return Ok((one - fy) * lower + fy * upper);
    }
    if method != 2 {
        return Err(ERR_DIMENSION);
    }

    let iy = y.floor() as isize;
    let ix = x.floor() as isize;
    let fy = y - iy as f32;
    let fx = x - ix as f32;
    let tiny = 1.0e-20f32;
    let mut rows = [0.0f32; 4];
    for (row_index, y_offset) in (-1isize..=2).enumerate() {
        let jy = (iy + y_offset).clamp(0, ny as isize - 1) as usize;
        let mut values = [0.0f32; 4];
        for (column_index, x_offset) in (-1isize..=2).enumerate() {
            let jx = (ix + x_offset).clamp(0, nx as isize - 1) as usize;
            let value = source_at(source, lead, jy, jx, ny, nx);
            values[column_index] = if value == 0.0 { tiny } else { value };
        }
        rows[row_index] = wps_oned(fx, values[0], values[1], values[2], values[3]);
    }
    let result = wps_oned(fy, rows[0], rows[1], rows[2], rows[3]);
    Ok(if result == tiny { 0.0 } else { result })
}

/// Interpolate one or more regular-grid FP32 fields to precomputed source
/// coordinates. `method` is 0=nearest, 1=bilinear, 2=WPS parabolic.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by the
/// dimensions.  Input and output buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_regular_interp_f32(
    source: *const f32,
    target_y: *const f32,
    target_x: *const f32,
    output: *mut f32,
    nlead: usize,
    source_ny: usize,
    source_nx: usize,
    ntarget: usize,
    method: i32,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || target_y.is_null() || target_x.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if nlead == 0
            || source_ny < 2
            || source_nx < 2
            || ntarget == 0
            || workers == 0
            || !(0..=2).contains(&method)
        {
            return ERR_DIMENSION;
        }
        let source_length = match nlead
            .checked_mul(source_ny)
            .and_then(|n| n.checked_mul(source_nx))
        {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let output_length = match nlead.checked_mul(ntarget) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let source_slice = std::slice::from_raw_parts(source, source_length);
        let y_slice = std::slice::from_raw_parts(target_y, ntarget);
        let x_slice = std::slice::from_raw_parts(target_x, ntarget);
        let output_address = output as usize;
        let error = AtomicI32::new(OK);
        std::thread::scope(|scope| {
            for (start, stop) in worker_ranges(ntarget, workers) {
                let error = &error;
                scope.spawn(move || {
                    let output_ptr = output_address as *mut f32;
                    for target_index in start..stop {
                        if error.load(Ordering::Relaxed) != OK {
                            break;
                        }
                        for lead in 0..nlead {
                            match horizontal_point(
                                source_slice,
                                lead,
                                y_slice[target_index],
                                x_slice[target_index],
                                source_ny,
                                source_nx,
                                method,
                            ) {
                                Ok(value) => unsafe {
                                    *output_ptr.add(lead * ntarget + target_index) = value;
                                },
                                Err(code) => {
                                    error
                                        .compare_exchange(
                                            OK,
                                            code,
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                        )
                                        .ok();
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });
        let code = error.load(Ordering::Relaxed);
        if code == OK
            && std::slice::from_raw_parts(output, output_length)
                .iter()
                .any(|value| !value.is_finite())
        {
            ERR_NONFINITE
        } else {
            code
        }
    }))
    .unwrap_or(ERR_PANIC)
}

#[inline]
fn lagrange(x: &[f32], y: &[f32], order: usize, target: f32) -> f32 {
    let mut result = 0.0f32;
    for term in 0..=order {
        let mut numerator = 1.0f32;
        let mut denominator = 1.0f32;
        for index in 0..=order {
            if index == term {
                continue;
            }
            numerator *= target - x[index];
            denominator *= x[term] - x[index];
        }
        if denominator != 0.0 {
            result += y[term] * numerator / denominator;
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn vertical_column(
    field: &[f32],
    surface_field: &[f32],
    source_pressure: &[f32],
    surface_pressure: &[f32],
    target_pressure: &[f32],
    output_address: usize,
    nsource: usize,
    ntarget: usize,
    ncolumn: usize,
    column: usize,
    interp_in_logp: bool,
    extrap_temperature: bool,
    force_surface: usize,
    zap_close_levels: f32,
    vboundb: usize,
) -> Result<(), i32> {
    let psfc = surface_pressure[column];
    if !psfc.is_finite() || psfc <= 0.0 || !surface_field[column].is_finite() {
        return Err(ERR_NONFINITE);
    }
    let mut previous = f32::INFINITY;
    let mut first_above = None;
    for level in 0..nsource {
        let pressure = source_pressure[level * ncolumn + column];
        let value = field[level * ncolumn + column];
        if !pressure.is_finite() || pressure <= 0.0 || !value.is_finite() {
            return Err(ERR_NONFINITE);
        }
        if level > 0 && pressure >= previous {
            return Err(ERR_PRESSURE_ORDER);
        }
        previous = pressure;
        if first_above.is_none() && pressure < psfc {
            first_above = Some(level);
        }
    }
    let first_above = first_above.ok_or(ERR_SURFACE_BRACKET)?;
    let mut ox = Vec::with_capacity(nsource + 1);
    let mut oy = Vec::with_capacity(nsource + 1);
    if first_above > 0 {
        for level in 0..first_above {
            ox.push(source_pressure[level * ncolumn + column]);
            oy.push(field[level * ncolumn + column]);
        }
        if ox[ox.len() - 1] - psfc < zap_close_levels {
            ox.pop();
            oy.pop();
        }
        ox.push(psfc);
        oy.push(surface_field[column]);
        let mut next = first_above;
        if force_surface > 0 {
            let force_pressure = target_pressure[(force_surface - 1) * ncolumn + column];
            for level in first_above..nsource {
                if source_pressure[level * ncolumn + column] <= force_pressure {
                    next = level;
                    break;
                }
            }
        }
        let start =
            if ox[ox.len() - 1] - source_pressure[next * ncolumn + column] < zap_close_levels {
                next + 1
            } else {
                next
            };
        for level in start..nsource {
            ox.push(source_pressure[level * ncolumn + column]);
            oy.push(field[level * ncolumn + column]);
        }
    } else {
        ox.push(psfc);
        oy.push(surface_field[column]);
        let mut next = 0usize;
        if force_surface > 0 {
            let force_pressure = target_pressure[(force_surface - 1) * ncolumn + column];
            for level in 0..nsource {
                if source_pressure[level * ncolumn + column] <= force_pressure {
                    next = level;
                    break;
                }
            }
        }
        for level in next..nsource {
            let pressure = source_pressure[level * ncolumn + column];
            if ox[ox.len() - 1] - pressure < zap_close_levels && level < nsource - 1 {
                continue;
            }
            ox.push(pressure);
            oy.push(field[level * ncolumn + column]);
        }
    }
    if ox.len() < 2 {
        return Err(ERR_INTERPOLATION_WINDOW);
    }
    let x: Vec<f32> = if interp_in_logp {
        ox.iter().map(|value| value.ln()).collect()
    } else {
        ox.clone()
    };
    let output = output_address as *mut f32;
    for target_level in 0..ntarget {
        let pressure = target_pressure[target_level * ncolumn + column];
        if !pressure.is_finite() || pressure <= 0.0 {
            return Err(ERR_NONFINITE);
        }
        let target_x = if interp_in_logp {
            pressure.ln()
        } else {
            pressure
        };
        let mut found = None;
        for lower in 0..x.len() - 1 {
            if (target_x - x[lower]) * (target_x - x[lower + 1]) <= 0.0 {
                found = Some(lower);
                break;
            }
        }
        let result = if let Some(lower) = found {
            if target_level + 1 >= 1 + vboundb {
                let fits_upper = lower + 2 <= x.len() - 1;
                let fits_lower = lower >= 1;
                if fits_upper && fits_lower {
                    0.5 * (lagrange(&x[lower..lower + 3], &oy[lower..lower + 3], 2, target_x)
                        + lagrange(
                            &x[lower - 1..lower + 2],
                            &oy[lower - 1..lower + 2],
                            2,
                            target_x,
                        ))
                } else if fits_upper {
                    lagrange(&x[lower..lower + 3], &oy[lower..lower + 3], 2, target_x)
                } else if fits_lower {
                    lagrange(
                        &x[lower - 1..lower + 2],
                        &oy[lower - 1..lower + 2],
                        2,
                        target_x,
                    )
                } else {
                    return Err(ERR_INTERPOLATION_WINDOW);
                }
            } else {
                lagrange(&x[lower..lower + 2], &oy[lower..lower + 2], 1, target_x)
            }
        } else if pressure > ox[0] {
            if extrap_temperature {
                let t1 = oy[0] * (ox[0] / 100000.0).powf(0.2857143);
                let average_pressure = 0.5 * (pressure + ox[0]);
                let dhdp = 11880.516 * 0.1902632 * (average_pressure / 100.0).powf(0.1902632 - 1.0);
                let dt = dhdp * ((pressure - ox[0]) / 100.0) * 0.0065;
                (t1 + dt) * (100000.0 / pressure).powf(0.2857143)
            } else {
                oy[0]
            }
        } else {
            return Err(ERR_TARGET_ABOVE_TOP);
        };
        if !result.is_finite() {
            return Err(ERR_NONFINITE);
        }
        unsafe {
            *output.add(target_level * ncolumn + column) = result;
        }
    }
    Ok(())
}

/// Apply WRF-real vertical interpolation to independent FP32 columns.
/// Unlike the CUDA v1 kernel, source columns are dynamically sized.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by the
/// dimensions.  Input and output buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_vert_interp_f32(
    field: *const f32,
    surface_field: *const f32,
    source_pressure: *const f32,
    surface_pressure: *const f32,
    target_pressure: *const f32,
    output: *mut f32,
    nsource: usize,
    ntarget: usize,
    ncolumn: usize,
    interp_in_logp: i32,
    extrap_temperature: i32,
    force_surface: usize,
    zap_close_levels: f32,
    vboundb: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if field.is_null()
            || surface_field.is_null()
            || source_pressure.is_null()
            || surface_pressure.is_null()
            || target_pressure.is_null()
            || output.is_null()
        {
            return ERR_NULL;
        }
        if nsource < 2
            || ntarget == 0
            || ncolumn == 0
            || workers == 0
            || force_surface > ntarget
            || !zap_close_levels.is_finite()
            || zap_close_levels < 0.0
            || !(0..=1).contains(&interp_in_logp)
            || !(0..=1).contains(&extrap_temperature)
        {
            return ERR_DIMENSION;
        }
        let source_length = match nsource.checked_mul(ncolumn) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let target_length = match ntarget.checked_mul(ncolumn) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let field_slice = std::slice::from_raw_parts(field, source_length);
        let source_slice = std::slice::from_raw_parts(source_pressure, source_length);
        let surface_field_slice = std::slice::from_raw_parts(surface_field, ncolumn);
        let surface_pressure_slice = std::slice::from_raw_parts(surface_pressure, ncolumn);
        let target_slice = std::slice::from_raw_parts(target_pressure, target_length);
        let output_address = output as usize;
        let error = AtomicI32::new(OK);
        std::thread::scope(|scope| {
            for (start, stop) in worker_ranges(ncolumn, workers) {
                let error = &error;
                scope.spawn(move || {
                    for column in start..stop {
                        if error.load(Ordering::Relaxed) != OK {
                            break;
                        }
                        if let Err(code) = vertical_column(
                            field_slice,
                            surface_field_slice,
                            source_slice,
                            surface_pressure_slice,
                            target_slice,
                            output_address,
                            nsource,
                            ntarget,
                            ncolumn,
                            column,
                            interp_in_logp != 0,
                            extrap_temperature != 0,
                            force_surface,
                            zap_close_levels,
                            vboundb,
                        ) {
                            error
                                .compare_exchange(OK, code, Ordering::Relaxed, Ordering::Relaxed)
                                .ok();
                            break;
                        }
                    }
                });
            }
        });
        error.load(Ordering::Relaxed)
    }))
    .unwrap_or(ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn gpuwm_preprocess_cpu_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_partition_is_complete_and_stable() {
        assert_eq!(worker_ranges(10, 3), vec![(0, 4), (4, 7), (7, 10)]);
        assert_eq!(worker_ranges(3, 9), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn nearest_uses_ties_to_even() {
        assert_eq!(nearest_even_nonnegative(0.5), 0);
        assert_eq!(nearest_even_nonnegative(1.5), 2);
        assert_eq!(nearest_even_nonnegative(2.5), 2);
    }

    #[test]
    fn parabolic_zero_sentinel_matches_cuda_flush_to_zero() {
        // This is the reduced 700-hPa GFS RH stencil that exposed a 0 versus
        // 3.28849-percent CPU/CUDA branch split in the full native pipeline.
        let source = [
            8.4, 9.3, 10.2, 11.4, 1.1, 2.0, 3.4, 5.1, 0.0, 0.0, 0.0, 0.2, 2.2, 1.7, 1.2, 0.8,
        ];
        let actual = horizontal_point(&source, 0, 1.002_899_2, 1.937_393_2, 4, 4, 2)
            .expect("valid parabolic stencil");
        assert_eq!(actual.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn vertical_column_has_no_64_level_ceiling() {
        let nsource = 96usize;
        let ntarget = 137usize;
        let ncolumn = 1usize;
        let source_pressure: Vec<f32> = (0..nsource)
            .map(|level| 100000.0 - level as f32 * 900.0)
            .collect();
        let field: Vec<f32> = source_pressure
            .iter()
            .map(|pressure| 200.0 + pressure * 0.001)
            .collect();
        let surface_pressure = [100500.0f32];
        let surface_field = [300.5f32];
        let target_pressure: Vec<f32> = (0..ntarget)
            .map(|level| 100250.0 - level as f32 * 600.0)
            .collect();
        let mut output = vec![0.0f32; ntarget];
        let code = unsafe {
            gpuwm_wrf_vert_interp_f32(
                field.as_ptr(),
                surface_field.as_ptr(),
                source_pressure.as_ptr(),
                surface_pressure.as_ptr(),
                target_pressure.as_ptr(),
                output.as_mut_ptr(),
                nsource,
                ntarget,
                ncolumn,
                0,
                0,
                1,
                0.0,
                4,
                7,
            )
        };
        assert_eq!(code, OK);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
