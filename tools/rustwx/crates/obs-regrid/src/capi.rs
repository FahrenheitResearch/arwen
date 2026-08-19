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
    OBSREGRID_ABI_VERSION
}

/// The source-revision stamp, same contract as
/// `gpuwm_static_source_rev` and `gpuwm_ncwrite_source_rev`: read out of
/// the binary as bytes by the release cut, never executed.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_obsregrid_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

/// Copy the thread-local last error into `buf` (UTF-8, no NUL); returns
/// the full message length so a short buffer is detectable.
///
/// # Safety
/// `buf` must point to `cap` writable bytes, or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsregrid_last_error(buf: *mut u8, cap: usize) -> usize {
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
        Method::CellAverage => source_cells,
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
        Method::CellAverage => source_cells,
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
