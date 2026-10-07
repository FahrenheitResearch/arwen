//! The C ABI seam `gpuwm/obs_regrid_bridge.py` loads through
//! ctypes.
//!
//! Same discipline as `static-fields/src/capi/` and
//! `netcdf-writer/src/capi.rs`: cdylib + ctypes (no pyo3 -- one loading
//! discipline across every gpuwm bridge), positional C signatures
//! guarded by an ABI version probe, a thread-local last-error string,
//! and a contract-marker symbol (`gpuwm_obsregrid_build_plan`, listed in
//! `gpuwm/bridges.py` `BRIDGE_ABI_MARKERS`) naming the capability that
//! distinguishes this contract.
//!
//! No handle registry, unlike `static-fields`.  A remap plan is two
//! small arrays the Python dataclass already owns and publishes -- the
//! battery reads `source_index` and `reachable` for its receipt, and the
//! plan crosses process boundaries in evidence -- so the seam is
//! stateless: build writes the plan into caller buffers, apply reads it
//! back.  A registry would have made the plan opaque and put a lifetime
//! on the one object in this subsystem that exists to be inspected.
//!
//! Array data crosses as raw little-endian f64/i64/u8, exactly
//! `numpy.tobytes()` of a C-contiguous array.  Booleans cross as u8
//! because that is `numpy.bool_`'s memory layout.

use std::cell::RefCell;

use crate::plan::{Method, apply_plan, build_plan};

/// The seam's ABI version.  Bump when a signature changes shape, never
/// for a rebuild.
pub const OBSREGRID_ABI_VERSION: u32 = 1;

const OK: i32 = 0;
const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Turn a panic below this seam into the ABI's own refusal.
///
/// An unwind that reaches an `extern "C"` boundary aborts the process on the
/// edition this workspace pins, so a panic in a decoder would take the host
/// Python interpreter with it instead of returning the negative code and
/// last-error string this ABI documents.  Every entry point runs its body
/// through here, the same discipline `tools/grib1_bridge/src/lib.rs` applies
/// to its own exports.
pub(crate) fn guard<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            // `try_borrow_mut`, not `borrow_mut`: the panic may have come from
            // inside the last-error accessor itself, and a second panic here
            // would abort exactly what this guard exists to prevent.
            LAST_ERROR.with(|slot| {
                if let Ok(mut message) = slot.try_borrow_mut() {
                    *message = format!("panic in the obs-regrid seam: {detail}");
                }
            });
            on_panic
        }
    }
}

/// # Safety
/// `ptr` must point to `len` readable `T`, or be null when `len` is 0.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr` must point to `len` writable `T`, or be null when `len` is 0.
unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]> {
    if len == 0 {
        return Some(&mut []);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_obsregrid_abi_version() -> u32 {
    guard(0, || {
        OBSREGRID_ABI_VERSION
    })
}

/// The source-revision stamp, same contract as
/// `gpuwm_static_source_rev` and `gpuwm_ncwrite_source_rev`: read out of
/// the binary as bytes by the release cut, never executed.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_obsregrid_source_rev() -> *const std::os::raw::c_char {
    guard(std::ptr::null(), || {
        static SOURCE_REV_STAMP: &str = concat!(
            "GPUWM_BRIDGE_SOURCE_REV=",
            env!("GPUWM_BRIDGE_SOURCE_REV"),
            "\0"
        );
        SOURCE_REV_STAMP.as_ptr().cast()
    })
}

/// Copy the thread-local last error into `buf` (UTF-8, no NUL); returns
/// the full message length so a short buffer is detectable.
///
/// # Safety
/// `buf` must point to `cap` writable bytes, or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_last_error(buf: *mut u8, cap: usize) -> usize {
    guard(0, || {
        LAST_ERROR.with(|slot| {
            let message = slot.borrow();
            let raw = message.as_bytes();
            if !buf.is_null() && cap > 0 {
                let n = raw.len().min(cap);
                unsafe {
                    std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n);
                }
            }
            raw.len()
        })
    })
}

/// Build a remap plan.
///
/// `method` is 0 for nearest and 1 for cell_average.  `out_source_index`
/// holds one i64 per DESTINATION cell for nearest and one per SOURCE
/// cell for cell_average; `out_reachable` holds one u8 per destination
/// cell in both cases.
///
/// # Safety
/// Every pointer must address the number of elements the shapes imply.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_obsregrid_build_plan(
    method: u32,
    source_latitude: *const f64,
    source_longitude: *const f64,
    source_ny: usize,
    source_nx: usize,
    destination_latitude: *const f64,
    destination_longitude: *const f64,
    destination_ny: usize,
    destination_nx: usize,
    max_distance_m: f64,
    out_source_index: *mut i64,
    out_reachable: *mut u8,
    out_max_used_distance_m: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let method = match Method::from_code(method) {
            Ok(value) => value,
            Err(error) => return set_error(error.to_string()),
        };
        let source_cells = match source_ny.checked_mul(source_nx) {
            Some(value) if value > 0 => value,
            _ => return set_error("the source grid must be a non-empty 2-D shape"),
        };
        let destination_cells = match destination_ny.checked_mul(destination_nx) {
            Some(value) if value > 0 => value,
            _ => return set_error("the destination grid must be a non-empty 2-D shape"),
        };
        let index_cells = match method {
            Method::Nearest => destination_cells,
            Method::CellAverage | Method::CellSum => source_cells,
            Method::CellSumSplit { n } => source_cells * n * n,
        };
        let (Some(source_latitude), Some(source_longitude)) = (
            unsafe { slice(source_latitude, source_cells) },
            unsafe { slice(source_longitude, source_cells) },
        ) else {
            return set_error("null source latitude/longitude pointer");
        };
        let (Some(destination_latitude), Some(destination_longitude)) = (
            unsafe { slice(destination_latitude, destination_cells) },
            unsafe { slice(destination_longitude, destination_cells) },
        ) else {
            return set_error("null destination latitude/longitude pointer");
        };
        let (Some(index_out), Some(reachable_out)) = (
            unsafe { slice_mut(out_source_index, index_cells) },
            unsafe { slice_mut(out_reachable, destination_cells) },
        ) else {
            return set_error("null plan output pointer");
        };
        if out_max_used_distance_m.is_null() {
            return set_error("null max-used-distance pointer");
        }

        let plan = match build_plan(
            method,
            source_latitude,
            source_longitude,
            (source_ny, source_nx),
            destination_latitude,
            destination_longitude,
            (destination_ny, destination_nx),
            max_distance_m,
        ) {
            Ok(value) => value,
            Err(error) => return set_error(error.to_string()),
        };
        index_out.copy_from_slice(&plan.source_index);
        for (slot, value) in reachable_out.iter_mut().zip(plan.reachable.iter()) {
            *slot = u8::from(*value);
        }
        unsafe {
            *out_max_used_distance_m = plan.max_used_distance_m;
        }
        OK
    })
}

/// Build a cell-sum plan, without changing the ABI-1 observation symbols.
/// # Safety
/// Output indices address source_ny * source_nx * n * n elements.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_obsregrid_build_sum_plan(
    method: u32,
    n: usize,
    source_latitude: *const f64,
    source_longitude: *const f64,
    source_ny: usize,
    source_nx: usize,
    destination_latitude: *const f64,
    destination_longitude: *const f64,
    destination_ny: usize,
    destination_nx: usize,
    max_distance_m: f64,
    out_source_index: *mut i64,
    out_reachable: *mut u8,
    out_max_used_distance_m: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let method = match (method, n) {
            (2, 1) => Method::CellSum,
            (3, _) => Method::CellSumSplit { n },
            _ => return set_error("sum builder requires method 2 with n=1 or method 3 with a positive partition count; other method layouts would overwrite the index buffer"),
        };
        if let Err(error) = crate::plan::split_count(method) { return set_error(error.to_string()); }
        let source_cells = match source_ny.checked_mul(source_nx) {
            Some(value) if value > 0 => value,
            _ => return set_error("the source grid must be a non-empty 2-D shape; otherwise mass assignment would read an invalid buffer"),
        };
        let destination_cells = match destination_ny.checked_mul(destination_nx) {
            Some(value) if value > 0 => value,
            _ => return set_error("the destination grid must be a non-empty 2-D shape; otherwise mass assignment would write an invalid buffer"),
        };
        let index_cells = match source_cells.checked_mul(n).and_then(|v| v.checked_mul(n)) {
            Some(v) if v <= crate::plan::MAX_SUM_INDEX_BYTES / 8 => v,
            None => return set_error("cell-sum plan size overflows; destination index buffer cannot be sized"),
            _ => return set_error("cell-sum destination indices exceed 4 GiB per buffer; caller and Rust copies would exhaust the supported host-memory budget"),
        };
        let (Some(source_latitude), Some(source_longitude)) = (
            unsafe { slice(source_latitude, source_cells) },
            unsafe { slice(source_longitude, source_cells) },
        ) else {
            return set_error("null source latitude/longitude pointer; refusing null coordinate reads");
        };
        let (Some(destination_latitude), Some(destination_longitude)) = (
            unsafe { slice(destination_latitude, destination_cells) },
            unsafe { slice(destination_longitude, destination_cells) },
        ) else {
            return set_error("null destination latitude/longitude pointer; refusing null coordinate reads");
        };
        let (Some(index_out), Some(reachable_out)) = (
            unsafe { slice_mut(out_source_index, index_cells) },
            unsafe { slice_mut(out_reachable, destination_cells) },
        ) else {
            return set_error("null plan output pointer; refusing null index writes");
        };
        if out_max_used_distance_m.is_null() {
            return set_error("null max-used-distance pointer; refusing null receipt writes");
        }

        let plan = match build_plan(
            method,
            source_latitude,
            source_longitude,
            (source_ny, source_nx),
            destination_latitude,
            destination_longitude,
            (destination_ny, destination_nx),
            max_distance_m,
        ) {
            Ok(value) => value,
            Err(error) => return set_error(error.to_string()),
        };
        index_out.copy_from_slice(&plan.source_index);
        for (slot, value) in reachable_out.iter_mut().zip(plan.reachable.iter()) {
            *slot = u8::from(*value);
        }
        unsafe {
            *out_max_used_distance_m = plan.max_used_distance_m;
        }
        OK
    })
}

/// Apply a plan to one field and its validity.
///
/// # Safety
/// Every pointer must address the number of elements the shapes imply.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_obsregrid_apply_plan(
    method: u32,
    source_index: *const i64,
    reachable: *const u8,
    source_ny: usize,
    source_nx: usize,
    destination_ny: usize,
    destination_nx: usize,
    values: *const f64,
    valid: *const u8,
    out_values: *mut f64,
    out_valid: *mut u8,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let method = match Method::from_code(method) {
            Ok(value) => value,
            Err(error) => return set_error(error.to_string()),
        };
        let source_cells = match source_ny.checked_mul(source_nx) {
            Some(value) if value > 0 => value,
            _ => return set_error("the source grid must be a non-empty 2-D shape"),
        };
        let destination_cells = match destination_ny.checked_mul(destination_nx) {
            Some(value) if value > 0 => value,
            _ => return set_error("the destination grid must be a non-empty 2-D shape"),
        };
        let index_cells = match method {
            Method::Nearest => destination_cells,
            Method::CellAverage | Method::CellSum => source_cells,
            Method::CellSumSplit { n } => source_cells * n * n,
        };
        let (Some(source_index), Some(reachable), Some(values), Some(valid)) = (
            unsafe { slice(source_index, index_cells) },
            unsafe { slice(reachable, destination_cells) },
            unsafe { slice(values, source_cells) },
            unsafe { slice(valid, source_cells) },
        ) else {
            return set_error("null plan or field pointer");
        };
        let (Some(out_values), Some(out_valid)) = (
            unsafe { slice_mut(out_values, destination_cells) },
            unsafe { slice_mut(out_valid, destination_cells) },
        ) else {
            return set_error("null output pointer");
        };

        let reachable: Vec<bool> = reachable.iter().map(|byte| *byte != 0).collect();
        let valid_field: Vec<bool> = valid.iter().map(|byte| *byte != 0).collect();
        let mut valid_out = vec![false; destination_cells];
        if let Err(error) = apply_plan(
            method,
            source_index,
            &reachable,
            (source_ny, source_nx),
            (destination_ny, destination_nx),
            values,
            &valid_field,
            out_values,
            &mut valid_out,
        ) {
            return set_error(error.to_string());
        }
        for (slot, value) in out_valid.iter_mut().zip(valid_out.iter()) {
            *slot = u8::from(*value);
        }
        OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_abi_probe_answers_the_declared_version() {
        assert_eq!(gpuwm_obsregrid_abi_version(), OBSREGRID_ABI_VERSION);
    }

    #[test]
    fn a_refusal_reaches_the_caller_through_the_error_slot() {
        let mut index = [0i64; 1];
        let mut reachable = [0u8; 1];
        let mut used = 0.0f64;
        let rc = unsafe {
            gpuwm_obsregrid_build_plan(
                0,
                [38.0f64].as_ptr(),
                [-98.0f64].as_ptr(),
                1,
                1,
                [38.0f64].as_ptr(),
                [-98.0f64].as_ptr(),
                1,
                1,
                // A bound that is not a distance.
                0.0,
                index.as_mut_ptr(),
                reachable.as_mut_ptr(),
                &mut used,
            )
        };
        assert_eq!(rc, ERR);
        let mut buffer = [0u8; 256];
        let length = unsafe { gpuwm_obsregrid_last_error(buffer.as_mut_ptr(), buffer.len()) };
        let message = std::str::from_utf8(&buffer[..length]).unwrap();
        assert!(message.contains("positive and finite"), "{message}");
    }

    #[test]
    fn a_null_output_pointer_is_refused_rather_than_written_through() {
        let mut reachable = [0u8; 1];
        let mut used = 0.0f64;
        let rc = unsafe {
            gpuwm_obsregrid_build_plan(
                0,
                [38.0f64].as_ptr(),
                [-98.0f64].as_ptr(),
                1,
                1,
                [38.0f64].as_ptr(),
                [-98.0f64].as_ptr(),
                1,
                1,
                50_000.0,
                std::ptr::null_mut(),
                reachable.as_mut_ptr(),
                &mut used,
            )
        };
        assert_eq!(rc, ERR);
    }

    #[test]
    fn build_then_apply_round_trips_through_the_seam() {
        let source_lat = [38.0f64, 38.1];
        let source_lon = [-98.0f64, -98.0];
        let mut index = [0i64; 2];
        let mut reachable = [0u8; 2];
        let mut used = 0.0f64;
        let rc = unsafe {
            gpuwm_obsregrid_build_plan(
                0,
                source_lat.as_ptr(),
                source_lon.as_ptr(),
                2,
                1,
                source_lat.as_ptr(),
                source_lon.as_ptr(),
                2,
                1,
                50_000.0,
                index.as_mut_ptr(),
                reachable.as_mut_ptr(),
                &mut used,
            )
        };
        assert_eq!(rc, OK);
        assert_eq!(reachable, [1, 1]);

        let mut out_values = [0.0f64; 2];
        let mut out_valid = [0u8; 2];
        let rc = unsafe {
            gpuwm_obsregrid_apply_plan(
                0,
                index.as_ptr(),
                reachable.as_ptr(),
                2,
                1,
                2,
                1,
                [4.5f64, 6.5].as_ptr(),
                [1u8, 0].as_ptr(),
                out_values.as_mut_ptr(),
                out_valid.as_mut_ptr(),
            )
        };
        assert_eq!(rc, OK);
        assert_eq!(out_values, [4.5, 0.0]);
        assert_eq!(out_valid, [1, 0]);
    }
}

/// Apply a cell-sum plan and publish its four mass totals.
/// # Safety
/// index has source_count*n*n elements; receipt has four f64 elements.
/// Other pointers address their explicit counts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_apply_sum_plan(
    n: usize, index: *const i64, source_count: usize, destination_count: usize,
    values: *const f64, valid: *const u8, out: *mut f64, out_valid: *mut u8,
    receipt: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let count = match source_count.checked_mul(n).and_then(|v| v.checked_mul(n)) {
            Some(v) if n > 0 => v,
            _ => return set_error("cell-sum split size is zero or overflows; sub-point buffer cannot be read"),
        };
        let (Some(index), Some(values), Some(valid), Some(out), Some(out_valid), Some(receipt)) = (
            unsafe { slice(index, count) }, unsafe { slice(values, source_count) },
            unsafe { slice(valid, source_count) }, unsafe { slice_mut(out, destination_count) },
            unsafe { slice_mut(out_valid, destination_count) }, unsafe { slice_mut(receipt, 4) },
        ) else { return set_error("null cell-sum buffer; refusing to read or write through null"); };
        let valid: Vec<bool> = valid.iter().map(|v| *v != 0).collect();
        let mut validity = vec![true; destination_count];
        match crate::plan::apply_sum(Method::CellSumSplit { n }, index, values, &valid, out, &mut validity) {
            Ok(totals) => receipt.copy_from_slice(&totals),
            Err(error) => return set_error(error.to_string()),
        }
        for (to, from) in out_valid.iter_mut().zip(validity) { *to = u8::from(from); }
        OK
    })
}

/// Apply a cell-sum plan to a fire-intensity field with the touch rule
/// (`plan::apply_touch_sum`): each destination a source cell's sub-points
/// reach gets that cell's whole value once.  Same buffers and receipt layout
/// as `gpuwm_obsregrid_apply_sum_plan`; receipt[1] is the touched total.
/// # Safety
/// index has source_count*n*n elements; receipt has four f64 elements.
/// Other pointers address their explicit counts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_apply_touch_plan(
    n: usize, index: *const i64, source_count: usize, destination_count: usize,
    values: *const f64, valid: *const u8, out: *mut f64, out_valid: *mut u8,
    receipt: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let count = match source_count.checked_mul(n).and_then(|v| v.checked_mul(n)) {
            Some(v) if n > 0 => v,
            _ => return set_error("cell-sum split size is zero or overflows; sub-point buffer cannot be read"),
        };
        let (Some(index), Some(values), Some(valid), Some(out), Some(out_valid), Some(receipt)) = (
            unsafe { slice(index, count) }, unsafe { slice(values, source_count) },
            unsafe { slice(valid, source_count) }, unsafe { slice_mut(out, destination_count) },
            unsafe { slice_mut(out_valid, destination_count) }, unsafe { slice_mut(receipt, 4) },
        ) else { return set_error("null cell-sum buffer; refusing to read or write through null"); };
        let valid: Vec<bool> = valid.iter().map(|v| *v != 0).collect();
        let mut validity = vec![true; destination_count];
        match crate::plan::apply_touch_sum(Method::CellSumSplit { n }, index, values, &valid, out, &mut validity) {
            Ok(totals) => receipt.copy_from_slice(&totals),
            Err(error) => return set_error(error.to_string()),
        }
        for (to, from) in out_valid.iter_mut().zip(validity) { *to = u8::from(from); }
        OK
    })
}

/// Apply one table-declared inclusive range mask in Rust.
/// # Safety
/// Pointers address count elements. valid is updated in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_mask_range(
    values: *const f64, count: usize, minimum: f64, maximum: f64, valid: *mut u8,
    quantity: *const f64, nonzero_only: u8,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (Some(values), Some(valid), Some(quantity)) = (unsafe { slice(values, count) }, unsafe { slice_mut(valid, count) }, unsafe { slice(quantity, count) })
            else { return set_error("null emission mask buffer; refusing null data access"); };
        if minimum.is_nan() || maximum.is_nan() || minimum > maximum {
            return set_error("emission mask range is reversed or NaN; it would exclude cells unpredictably");
        }
        for ((value, valid), quantity) in values.iter().zip(valid).zip(quantity) {
            if nonzero_only != 0 && *quantity == 0.0 { continue; }
            *valid = u8::from(*valid != 0 && value.is_finite() && *value >= minimum && *value <= maximum);
        }
        OK
    })
}

/// Normalize a row-declared no-fire sentinel, never arbitrary missing values.
/// # Safety
/// values addresses count writable f64 elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_emission_sentinel(values: *mut f64, count: usize, sentinel: f64) -> i32 {
    guard(ERR, || {
        clear_error();
        let Some(values) = (unsafe { slice_mut(values, count) }) else {
            return set_error("null emission values; refusing null data access");
        };
        for value in values { if *value == sentinel { *value = 0.0; } }
        OK
    })
}

/// Select n from the largest source angular spacing and finest model spacing.
/// # Safety
/// map_factors addresses count readable f64 elements, out_n one usize.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_emission_split(
    latitude: *const f64, longitude: *const f64, source_ny: usize, source_nx: usize,
    dx: f64, map_factors: *const f64, count: usize, out_n: *mut usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let source_count = match source_ny.checked_mul(source_nx) {
            Some(v) if v > 0 => v,
            _ => return set_error("source shape is empty or overflows; emission spacing cannot be determined"),
        };
        let (Some(latitude), Some(longitude)) = (unsafe { slice(latitude, source_count) }, unsafe { slice(longitude, source_count) })
            else { return set_error("null source coordinates; emission partition spacing cannot be determined"); };
        let (dy, dx_deg) = match crate::plan::regular_spacing(latitude, longitude, (source_ny, source_nx)) {
            Ok(v) => v, Err(error) => return set_error(error.to_string()),
        };
        let spacing_deg = dy.abs().max(dx_deg.abs());
        let Some(factors) = (unsafe { slice(map_factors, count) }) else {
            return set_error("null map factors; cannot determine emission partition spacing");
        };
        if count == 0 || out_n.is_null() || !dx.is_finite() || dx <= 0.0
            || !spacing_deg.is_finite() || spacing_deg <= 0.0
            || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return set_error("emission split requires positive finite spacing and map factors; otherwise a coarse fire cell concentrates in one fine cell");
        }
        let largest = factors.iter().copied().fold(0.0, f64::max);
        let n = (crate::geometry::EARTH_RADIUS_M * spacing_deg.to_radians() * largest / dx).ceil().max(1.0);
        if !n.is_finite() || n >= usize::MAX as f64 { return set_error("emission split count cannot fit usize; sub-point buffers cannot be represented"); }
        unsafe { *out_n = n as usize; }
        OK
    })
}
