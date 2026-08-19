//! The C ABI seam `gpuwm/static/rust_bridge.py` loads through ctypes.
//!
//! Same discipline as `netcdf-writer/src/capi.rs`: cdylib + ctypes (no
//! pyo3 -- one loading discipline across every gpuwm bridge), positional
//! C signatures guarded by an ABI version probe, a thread-local
//! last-error string, and a contract-marker symbol
//! (`gpuwm_static_build_fields`, listed in `gpuwm/bridges.py`
//! `BRIDGE_ABI_MARKERS`) naming the capability that distinguishes this
//! contract.
//!
//! Handle model: opaque `u64` handles into process-global registries --
//! grids ([`grid`]) and field sets ([`build`]) -- because grids are
//! built once and queried many times, and field sets are large enough
//! that the Python side must copy plane-by-plane into preallocated
//! numpy buffers rather than round-trip JSON.  Scalars/config cross as
//! UTF-8 JSON; array data crosses as raw little-endian f64, exactly
//! `numpy.tobytes()` of a C-contiguous array.
//!
//! Entry-point ownership (files, not fences): [`grid`] is lane 1,
//! [`build`] is lane 2, [`highres`] is lane 3.  This module owns only
//! the shared plumbing below.

pub mod build;
pub mod grid;
pub mod highres;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::projection::ProjectedGrid;
use crate::types::FieldSet;

/// The seam's ABI version.  Bump when a signature changes shape, never
/// for a rebuild.
pub const STATIC_ABI_VERSION: u32 = 1;

pub(crate) const OK: i32 = 0;
pub(crate) const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(crate) fn set_error(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

pub(crate) fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn next_handle(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed) + 1
}

static GRID_COUNTER: AtomicU64 = AtomicU64::new(0);
static GRIDS: Mutex<BTreeMap<u64, ProjectedGrid>> = Mutex::new(BTreeMap::new());

static FIELDSET_COUNTER: AtomicU64 = AtomicU64::new(0);
static FIELDSETS: Mutex<BTreeMap<u64, FieldSet>> = Mutex::new(BTreeMap::new());

pub(crate) fn register_grid(grid: ProjectedGrid) -> u64 {
    let handle = next_handle(&GRID_COUNTER);
    GRIDS.lock().expect("grid registry poisoned").insert(handle, grid);
    handle
}

pub(crate) fn with_grid<T>(
    handle: u64,
    f: impl FnOnce(&ProjectedGrid) -> T,
) -> Option<T> {
    GRIDS.lock().expect("grid registry poisoned").get(&handle).map(f)
}

pub(crate) fn drop_grid(handle: u64) -> bool {
    GRIDS.lock().expect("grid registry poisoned").remove(&handle).is_some()
}

pub(crate) fn register_fieldset(fields: FieldSet) -> u64 {
    let handle = next_handle(&FIELDSET_COUNTER);
    FIELDSETS
        .lock()
        .expect("fieldset registry poisoned")
        .insert(handle, fields);
    handle
}

pub(crate) fn with_fieldset<T>(
    handle: u64,
    f: impl FnOnce(&FieldSet) -> T,
) -> Option<T> {
    FIELDSETS
        .lock()
        .expect("fieldset registry poisoned")
        .get(&handle)
        .map(f)
}

pub(crate) fn drop_fieldset(handle: u64) -> bool {
    FIELDSETS
        .lock()
        .expect("fieldset registry poisoned")
        .remove(&handle)
        .is_some()
}

/// # Safety
/// `ptr` must point to `len` readable bytes, or be null when `len` is 0.
pub(crate) unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr`/`len` as for [`bytes`]; the bytes must be UTF-8.
pub(crate) unsafe fn utf8<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    let raw = unsafe { bytes(ptr, len) }?;
    std::str::from_utf8(raw).ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_abi_version() -> u32 {
    STATIC_ABI_VERSION
}

/// The source-revision stamp, same contract as
/// `gpuwm_ncwrite_source_rev` (see `netcdf-writer`): read out of the
/// binary as bytes by the release cut, never executed.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

/// Copy the thread-local last error into `buf` (UTF-8, no NUL);
/// returns the full message length so a short buffer is detectable.
///
/// # Safety
/// `buf` must point to `cap` writable bytes, or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_last_error(
    buf: *mut u8,
    cap: usize,
) -> usize {
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
