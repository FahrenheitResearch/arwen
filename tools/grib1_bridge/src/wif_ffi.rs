//! C ABI for the two operators the static-dataset ingest route needed and
//! this library did not yet have.
//!
//! Neither entry knows anything about aerosols.  They are the two GENERIC
//! gaps the WIF aerosol climatology exposed:
//!
//! 1. **Decode.** `gpuwm_regular_interp_f32` and friends start from arrays
//!    that were already in memory.  Nothing in this library could READ a
//!    WPS intermediate file, even though `met_intermediate` in this same
//!    package WRITES one.  `wps_intermediate.rs` closes that, and these
//!    two entries expose it.  Any IFV=5 file works -- ungrib's output, our
//!    own writer's output, or a `constants_name` static dataset.
//!
//! 2. **Cyclic horizontal.** `horizontal_point`'s bilinear derives its
//!    donor with an FP32 floor and clamps `ix` to `nx - 2`, which is right
//!    for a bounded regional source and WRONG at the seam of a GLOBAL
//!    lat-lon source: a target between the last and first columns must
//!    take its second donor from column 0, not from column `nx - 1`
//!    twice.  `gpuwm_regular_cyclic_bilinear_f32` is the global-source
//!    operator, and it decides cyclicity from the file's OWN declared axis
//!    span rather than from a caller flag -- metadata, not a switch.
//!
//! Its arithmetic is the tree's existing
//! `canonical-f32-coordinate-f64-bilinear-single-round-v1` policy
//! (`gpuwm/ingest/preprocess_backend.py:PSFC_MAPPING_POLICY`): source
//! coordinates and bilinear weights in FP64, the four-term sum in FP64,
//! ONE round to FP32 at the end.  That is the arithmetic the aerosol
//! reference was oracled against real.exe with, so adopting it here keeps
//! the ported numbers and moves only the language.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::wps_intermediate;
use crate::{worker_ranges, ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};

/// The input file could not be opened or read.
pub const ERR_INPUT_OPEN: i32 = 8;
/// The input file's record structure is not the declared format.
pub const ERR_INPUT_FORMAT: i32 = 9;
/// A target point lies outside the source grid in a direction the source
/// is not cyclic in.
pub const ERR_TARGET_OFF_GRID: i32 = 10;

/// Fixed metadata stride, in f64 slots, of `gpuwm_wps_intermediate_read`'s
/// `meta_out`: xlvl, nx, ny, iproj, startlat, startlon, deltalat, deltalon.
pub const WPS_META_STRIDE: usize = 8;
/// Fixed byte stride of `names_out`: the format's own `field*9`.
pub const WPS_NAME_STRIDE: usize = 9;

fn last_error() -> &'static Mutex<String> {
    static SLOT: OnceLock<Mutex<String>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(String::new()))
}

fn set_last_error(message: String) {
    if let Ok(mut slot) = last_error().lock() {
        *slot = message;
    }
}

/// Copy the message behind the most recent nonzero return of a
/// `gpuwm_wps_intermediate_*` call into `buffer`, returning the byte count
/// written (never more than `capacity`).
///
/// An integer code cannot say WHICH version a rejected file declared or
/// which record ran short, and a refusal that cannot name its breakage is
/// not a refusal.  This is how the Python side gets the sentence.
///
/// # Safety
///
/// `buffer` must address `capacity` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_bridge_last_error(buffer: *mut u8, capacity: usize) -> usize {
    if buffer.is_null() || capacity == 0 {
        return 0;
    }
    let message = match last_error().lock() {
        Ok(slot) => slot.clone(),
        Err(_) => String::new(),
    };
    let bytes = message.as_bytes();
    let count = bytes.len().min(capacity);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, count);
    count
}

unsafe fn path_from(path: *const u8, path_len: usize) -> Option<PathBuf> {
    if path.is_null() || path_len == 0 {
        return None;
    }
    let bytes = std::slice::from_raw_parts(path, path_len);
    std::str::from_utf8(bytes).ok().map(PathBuf::from)
}

fn classify(error: &wps_intermediate::ReadError) -> i32 {
    match error {
        wps_intermediate::ReadError::Io(_) => ERR_INPUT_OPEN,
        _ => ERR_INPUT_FORMAT,
    }
}

/// Header-only pass over a WPS intermediate file: how many field records
/// it holds and how many data points in total, so the caller can size the
/// buffers `gpuwm_wps_intermediate_read` fills.
///
/// # Safety
///
/// `path` must address `path_len` readable bytes of UTF-8; the two output
/// pointers must each address a writable `u64`.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wps_intermediate_inventory(
    path: *const u8,
    path_len: usize,
    records_out: *mut u64,
    points_out: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if records_out.is_null() || points_out.is_null() {
            return ERR_NULL;
        }
        let Some(path) = path_from(path, path_len) else {
            set_last_error("the input path is empty or not valid UTF-8".to_string());
            return ERR_NULL;
        };
        match wps_intermediate::inventory(&path) {
            Ok((records, points)) => {
                *records_out = records as u64;
                *points_out = points as u64;
                OK
            }
            Err(error) => {
                let code = classify(&error);
                set_last_error(format!("{}: {error}", path.display()));
                code
            }
        }
    }))
    .unwrap_or(ERR_PANIC)
}

/// Read every field record of a WPS intermediate file into caller buffers
/// sized by `gpuwm_wps_intermediate_inventory`.
///
/// `names_out` receives `WPS_NAME_STRIDE` bytes per record (space-padded,
/// exactly the format's own `field*9`); `meta_out` receives
/// `WPS_META_STRIDE` f64 per record; `data_out` receives every record's
/// `ny*nx` values concatenated in file order, x fastest.
///
/// # Safety
///
/// Every pointer must address the complete buffer implied by `n_records`
/// and `n_points`; the buffers must not overlap.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_wps_intermediate_read(
    path: *const u8,
    path_len: usize,
    names_out: *mut u8,
    meta_out: *mut f64,
    data_out: *mut f32,
    n_records: u64,
    n_points: u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if names_out.is_null() || meta_out.is_null() || data_out.is_null() {
            return ERR_NULL;
        }
        let Some(path) = path_from(path, path_len) else {
            set_last_error("the input path is empty or not valid UTF-8".to_string());
            return ERR_NULL;
        };
        let (metas, data) = match wps_intermediate::read_all(&path) {
            Ok(value) => value,
            Err(error) => {
                let code = classify(&error);
                set_last_error(format!("{}: {error}", path.display()));
                return code;
            }
        };
        if metas.len() as u64 != n_records || data.len() as u64 != n_points {
            set_last_error(format!(
                "{}: the file changed between the inventory pass ({n_records} records, \
                 {n_points} points) and the read pass ({} records, {} points)",
                path.display(),
                metas.len(),
                data.len()
            ));
            return ERR_INPUT_FORMAT;
        }
        for (index, meta) in metas.iter().enumerate() {
            let name = meta.field.as_bytes();
            for slot in 0..WPS_NAME_STRIDE {
                let byte = if slot < name.len() { name[slot] } else { b' ' };
                *names_out.add(index * WPS_NAME_STRIDE + slot) = byte;
            }
            let base = meta_out.add(index * WPS_META_STRIDE);
            *base = f64::from(meta.xlvl);
            *base.add(1) = meta.nx as f64;
            *base.add(2) = meta.ny as f64;
            *base.add(3) = f64::from(meta.iproj);
            *base.add(4) = f64::from(meta.startlat);
            *base.add(5) = f64::from(meta.startlon);
            *base.add(6) = f64::from(meta.deltalat);
            *base.add(7) = f64::from(meta.deltalon);
        }
        std::ptr::copy_nonoverlapping(data.as_ptr(), data_out, data.len());
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// One target point of the global-capable bilinear operator.
///
/// Statement order is deliberate and load-bearing: it mirrors the NumPy
/// expression the aerosol reference was oracled with, term by term, so the
/// two agree to the last bit rather than "closely".
#[inline]
#[allow(clippy::too_many_arguments)]
fn cyclic_bilinear_point(
    source: &[f32],
    lead: usize,
    target_lat: f64,
    target_lon: f64,
    nlat: usize,
    nlon: usize,
    lat0: f64,
    dlat: f64,
    lon0: f64,
    dlon: f64,
    cyclic_x: bool,
) -> Result<f32, i32> {
    if !target_lat.is_finite() || !target_lon.is_finite() {
        return Err(ERR_TARGET_OFF_GRID);
    }
    let y = (target_lat - lat0) / dlat;
    let mut x = (target_lon - lon0) / dlon;
    if cyclic_x {
        x = x.rem_euclid(nlon as f64);
    }
    if y < 0.0 || y > (nlat - 1) as f64 {
        return Err(ERR_TARGET_OFF_GRID);
    }
    if !cyclic_x && (x < 0.0 || x > (nlon - 1) as f64) {
        return Err(ERR_TARGET_OFF_GRID);
    }
    // Donor selection is symmetric in the two axes: floor, then either
    // wrap (cyclic longitude) or clamp into the last full cell (latitude,
    // and a bounded longitude axis).  The clamp happens BEFORE the
    // fraction is taken, so a target exactly on the last row/column
    // interpolates within the final cell at fraction 1 instead of
    // reaching past the array.
    let y0 = (y.floor() as i64).clamp(0, nlat as i64 - 2);
    let x0 = if cyclic_x {
        x.floor() as i64
    } else {
        (x.floor() as i64).clamp(0, nlon as i64 - 2)
    };
    let fy = y - y0 as f64;
    let fx = x - x0 as f64;
    let x1 = x0 + 1;
    let (x0, x1) = if cyclic_x {
        (x0.rem_euclid(nlon as i64), x1.rem_euclid(nlon as i64))
    } else {
        (x0, x1)
    };
    let (y0, x0, x1) = (y0 as usize, x0 as usize, x1 as usize);
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w01 = fx * (1.0 - fy);
    let w10 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    let plane = lead * nlat * nlon;
    let at = |j: usize, i: usize| f64::from(source[plane + j * nlon + i]);
    let value = at(y0, x0) * w00 + at(y0, x1) * w01 + at(y0 + 1, x0) * w10 + at(y0 + 1, x1) * w11;
    Ok(value as f32)
}

/// Bilinear interpolation from a regular lat/lon source that may be
/// GLOBAL in longitude, to arbitrary target lat/lon points.
///
/// The source axes are reconstructed from the grid's OWN declared corner
/// and increment exactly as a file reader would (`axis[i] = start +
/// delta * i` in FP64, then `d = axis[1] - axis[0]`), and longitude
/// cyclicity is decided from the resulting span: a source whose
/// `nlon * dlon` is 360 degrees to within 1e-6 wraps, anything else is
/// bounded and a target outside it is refused rather than clamped.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by
/// the dimensions.  Input and output buffers must not overlap.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_regular_cyclic_bilinear_f32(
    source: *const f32,
    target_lat: *const f64,
    target_lon: *const f64,
    output: *mut f32,
    nlead: usize,
    source_ny: usize,
    source_nx: usize,
    ntarget: usize,
    startlat: f64,
    deltalat: f64,
    startlon: f64,
    deltalon: f64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || target_lat.is_null() || target_lon.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if nlead == 0 || source_ny < 2 || source_nx < 2 || ntarget == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        let Some(source_length) = nlead
            .checked_mul(source_ny)
            .and_then(|n| n.checked_mul(source_nx))
        else {
            return ERR_DIMENSION;
        };
        let Some(output_length) = nlead.checked_mul(ntarget) else {
            return ERR_DIMENSION;
        };
        // The axes exactly as a reader builds them, so `dlat`/`dlon` carry
        // the same FP64 value the reference's `latitude[1] - latitude[0]`
        // does -- which is not always `f64(deltalat)`.
        let lat0 = startlat;
        let lat1 = startlat + deltalat;
        let dlat = lat1 - lat0;
        let lon0 = startlon;
        let lon1 = startlon + deltalon;
        let dlon = lon1 - lon0;
        if dlat == 0.0 || dlon == 0.0 {
            return ERR_DIMENSION;
        }
        let span = source_nx as f64 * dlon;
        let cyclic_x = (span.abs() - 360.0).abs() < 1.0e-6;

        let source_slice = std::slice::from_raw_parts(source, source_length);
        let lat_slice = std::slice::from_raw_parts(target_lat, ntarget);
        let lon_slice = std::slice::from_raw_parts(target_lon, ntarget);
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
                            match cyclic_bilinear_point(
                                source_slice,
                                lead,
                                lat_slice[target_index],
                                lon_slice[target_index],
                                source_ny,
                                source_nx,
                                lat0,
                                dlat,
                                lon0,
                                dlon,
                                cyclic_x,
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
