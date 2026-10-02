//! LANE 2 seam: the static build and field-set access.
//!
//! `gpuwm_static_sampling_portable_v1` is this library's contract marker (see
//! `BRIDGE_ABI_MARKERS` in `gpuwm/bridges.py`): a build that loads and
//! answers the version probe but predates the field build cannot
//! produce a single static field, so the literal to check for is this
//! symbol name.

use super::{
    clear_error, drop_fieldset, guard, register_fieldset, set_error, utf8,
    with_fieldset, with_grid, ERR, OK,
};
use crate::fields::{build_static, GeogPaths};
use crate::HALO;

/// Build every native static field for a grid handle from nine resolved
/// GEOG dataset paths (JSON `GeogPaths`).  Writes a field-set handle.
/// Coverage receipts are queried per field afterwards.  LANE 2.
///
/// # Safety
/// `paths_json`/`paths_len` must describe readable UTF-8; `out_handle`
/// must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_build_fields(
    grid: u64,
    paths_json: *const u8,
    paths_len: usize,
    halo: u32,
    out_handle: *mut u64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let Some(text) = (unsafe { utf8(paths_json, paths_len) }) else {
            return set_error("GEOG paths pointer/UTF-8 invalid");
        };
        let paths: GeogPaths = match serde_json::from_str(text) {
            Ok(paths) => paths,
            Err(err) => return set_error(format!("GEOG paths JSON: {err}")),
        };
        let halo = if halo == u32::MAX { HALO } else { halo as usize };
        let built = with_grid(grid, |grid| build_static(grid, &paths, halo));
        match built {
            None => set_error(format!("unknown grid handle {grid}")),
            Some(Err(err)) => set_error(err.to_string()),
            Some(Ok(fields)) => {
                if out_handle.is_null() {
                    return set_error("out_handle is null");
                }
                unsafe { *out_handle = register_fieldset(fields) };
                OK
            }
        }
    })
}

/// Legacy orographic entry marker retained for ABI compatibility.
/// Corrected statistics are identified by `gpuwm_static_orographic_v2`.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_orographic_v1() -> u32 {
    1
}

/// WPS default-REAL coordinates, interpolation, and post-interpolation scaling.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_orographic_v2() -> u32 {
    2
}

/// Build WRF's sub-grid orographic statistics (VAR_SSO, CON, VAR, OA1-4,
/// OL1-4 and the GSL large- and small-scale sets) for a grid handle from a
/// JSON `OrographicRequest`.  Writes a field-set handle holding only them.
///
/// # Safety
/// `request_json`/`request_len` must describe readable UTF-8; `out_handle`
/// must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_build_orographic(
    grid: u64,
    request_json: *const u8,
    request_len: usize,
    halo: u32,
    out_handle: *mut u64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let Some(text) = (unsafe { utf8(request_json, request_len) }) else {
            return set_error("orographic request pointer/UTF-8 invalid");
        };
        let request: crate::fields::OrographicRequest =
            match serde_json::from_str(text) {
                Ok(request) => request,
                Err(err) => {
                    return set_error(format!("orographic request JSON: {err}"))
                }
            };
        let halo = if halo == u32::MAX { HALO } else { halo as usize };
        let built = with_grid(grid, |grid| {
            crate::fields::build_orographic(grid, &request, halo)
        });
        match built {
            None => set_error(format!("unknown grid handle {grid}")),
            Some(Err(err)) => set_error(err.to_string()),
            Some(Ok(fields)) => {
                if out_handle.is_null() {
                    return set_error("out_handle is null");
                }
                unsafe { *out_handle = register_fieldset(fields) };
                OK
            }
        }
    })
}

/// Build a terrain-only field set for the vertical survey.
/// # Safety
/// The path must be readable UTF-8 and out_handle must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_build_terrain(
    grid: u64, path: *const u8, path_len: usize, halo: u32, out_handle: *mut u64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let Some(path) = (unsafe { utf8(path, path_len) }) else {
            return set_error("terrain path pointer/UTF-8 invalid");
        };
        let halo = if halo == u32::MAX { HALO } else { halo as usize };
        match with_grid(grid, |g| crate::fields::build_terrain(g, std::path::Path::new(path), halo)) {
            None => set_error(format!("unknown grid handle {grid}")),
            Some(Err(err)) => set_error(err.to_string()),
            Some(Ok(fields)) => {
                if out_handle.is_null() { return set_error("out_handle is null"); }
                unsafe { *out_handle = register_fieldset(fields) };
                OK
            }
        }
    })
}

/// Number of fields in a set, or -1.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_fieldset_len(handle: u64) -> i64 {
    guard(ERR as i64, || {
        clear_error();
        match with_fieldset(handle, |set| set.fields.len() as i64) {
            Some(n) => n,
            None => {
                set_error(format!("unknown fieldset handle {handle}"));
                ERR as i64
            }
        }
    })
}

/// Name of field `index` (BTreeMap order, stable), copied UTF-8 into
/// `buf`; returns the full name length or -1.
///
/// # Safety
/// `buf` must point to `cap` writable bytes or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_fieldset_name(
    handle: u64,
    index: usize,
    buf: *mut u8,
    cap: usize,
) -> i64 {
    guard(ERR as i64, || {
        clear_error();
        let copied = with_fieldset(handle, |set| {
            let name = set.fields.keys().nth(index)?;
            let raw = name.as_bytes();
            if !buf.is_null() && cap > 0 {
                let n = raw.len().min(cap);
                unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n) };
            }
            Some(raw.len() as i64)
        });
        match copied {
            None => {
                set_error(format!("unknown fieldset handle {handle}"));
                ERR as i64
            }
            Some(None) => {
                set_error(format!("fieldset index {index} out of range"));
                ERR as i64
            }
            Some(Some(len)) => len,
        }
    })
}

/// Dims of one field by name: writes `(planes, ny, nx)` (planes == 1
/// for a 2-D field).
///
/// # Safety
/// `name`/`name_len` readable UTF-8; the three out pointers writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_field_dims(
    handle: u64,
    name: *const u8,
    name_len: usize,
    out_planes: *mut u64,
    out_ny: *mut u64,
    out_nx: *mut u64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let Some(name) = (unsafe { utf8(name, name_len) }) else {
            return set_error("field name pointer/UTF-8 invalid");
        };
        let dims = with_fieldset(handle, |set| {
            set.fields.get(name).map(crate::types::Field::dims)
        });
        match dims {
            None => set_error(format!("unknown fieldset handle {handle}")),
            Some(None) => set_error(format!("fieldset has no field {name:?}")),
            Some(Some((planes, ny, nx))) => {
                if out_planes.is_null() || out_ny.is_null() || out_nx.is_null() {
                    return set_error("dims out pointers are null");
                }
                unsafe {
                    *out_planes = planes as u64;
                    *out_ny = ny as u64;
                    *out_nx = nx as u64;
                }
                OK
            }
        }
    })
}

/// Copy one field's f64 data (C order, native endianness) into `out`.
///
/// # Safety
/// `name`/`name_len` readable UTF-8; `out` writable for `out_len` f64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_field_read(
    handle: u64,
    name: *const u8,
    name_len: usize,
    out: *mut f64,
    out_len: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let Some(name) = (unsafe { utf8(name, name_len) }) else {
            return set_error("field name pointer/UTF-8 invalid");
        };
        let result = with_fieldset(handle, |set| {
            let field = set.fields.get(name)?;
            let data = field.data();
            if data.len() != out_len || out.is_null() {
                return Some(Err(format!(
                    "field {name:?} has {} values, caller offered {out_len}",
                    data.len()
                )));
            }
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), out, out_len) };
            Some(Ok(()))
        });
        match result {
            None => set_error(format!("unknown fieldset handle {handle}")),
            Some(None) => set_error(format!("fieldset has no field {name:?}")),
            Some(Some(Err(message))) => set_error(message),
            Some(Some(Ok(()))) => OK,
        }
    })
}

/// Copy one field's source-coverage receipt JSON
/// (`gpuwm-geog-source-coverage-v1`); returns full length or -1.
///
/// # Safety
/// `name`/`name_len` readable UTF-8; `buf` writable for `cap` bytes or
/// null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_field_coverage_json(
    handle: u64,
    name: *const u8,
    name_len: usize,
    buf: *mut u8,
    cap: usize,
) -> i64 {
    guard(ERR as i64, || {
        clear_error();
        let Some(name) = (unsafe { utf8(name, name_len) }) else {
            set_error("field name pointer/UTF-8 invalid");
            return ERR as i64;
        };
        let copied = with_fieldset(handle, |set| {
            let report = set.coverage_reports.get(name)?;
            let raw = report.as_bytes();
            if !buf.is_null() && cap > 0 {
                let n = raw.len().min(cap);
                unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n) };
            }
            Some(raw.len() as i64)
        });
        match copied {
            None => {
                set_error(format!("unknown fieldset handle {handle}"));
                ERR as i64
            }
            Some(None) => {
                set_error(format!("no coverage receipt for field {name:?}"));
                ERR as i64
            }
            Some(Some(len)) => len,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_fieldset_free(handle: u64) {
    guard((), || {
        drop_fieldset(handle);
    })
}

// ---------------------------------------------------------------------------
// Terrain smoothing (additive; the entries above keep their signatures and
// STATIC_ABI_VERSION is unchanged, so a Python side that never asks for a
// non-default smoothing never binds these).  `smoothing` is the JSON echo
// `{"smooth_option": ..., "smooth_passes": ...}`, plus `"smooth_precision"`
// when a setting names one.
// ---------------------------------------------------------------------------

/// Present when the smoothing JSON below takes `smooth_precision`
/// (`"wps-float32"` on the default smoother).  The Python side asks for
/// this symbol before it sends the key, so a library staged before it is
/// named stale with its remedy instead of refusing an unknown field.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_terrain_smoothing_precision_v1() -> u32 {
    1
}

/// Parse the smoothing JSON handed over by pointer and length.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
unsafe fn smoothing_arg(
    ptr: *const u8,
    len: usize,
) -> std::result::Result<crate::smooth::TerrainSmoothing, String> {
    let Some(text) = (unsafe { utf8(ptr, len) }) else {
        return Err("terrain smoothing pointer/UTF-8 invalid".to_string());
    };
    crate::smooth::TerrainSmoothing::parse(text).map_err(|e| e.to_string())
}

/// `gpuwm_static_build_fields` with a terrain smoothing.
///
/// # Safety
/// `paths_json` and `smoothing_ptr` must address readable UTF-8 of the
/// given lengths; `out_handle` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_build_fields_smoothed(
    grid: u64,
    paths_json: *const u8,
    paths_len: usize,
    halo: u32,
    smoothing_ptr: *const u8,
    smoothing_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let smoothing = match unsafe { smoothing_arg(smoothing_ptr, smoothing_len) } {
            Ok(value) => value,
            Err(err) => return set_error(err),
        };
        let Some(text) = (unsafe { utf8(paths_json, paths_len) }) else {
            return set_error("GEOG paths pointer/UTF-8 invalid");
        };
        let paths: GeogPaths = match serde_json::from_str(text) {
            Ok(paths) => paths,
            Err(err) => return set_error(format!("GEOG paths JSON: {err}")),
        };
        let halo = if halo == u32::MAX { HALO } else { halo as usize };
        let built = with_grid(grid, |grid| {
            crate::fields::build_static_smoothed(grid, &paths, halo, smoothing)
        });
        match built {
            None => set_error(format!("unknown grid handle {grid}")),
            Some(Err(err)) => set_error(err.to_string()),
            Some(Ok(fields)) => {
                if out_handle.is_null() {
                    return set_error("out_handle is null");
                }
                unsafe { *out_handle = register_fieldset(fields) };
                OK
            }
        }
    })
}

/// Terrain alone (HGT_M) with a terrain smoothing, for the vertical survey.
///
/// # Safety
/// `path` and `smoothing_ptr` must address readable UTF-8 of the given
/// lengths; `out_handle` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_build_terrain_smoothed(
    grid: u64,
    path: *const u8,
    path_len: usize,
    halo: u32,
    smoothing_ptr: *const u8,
    smoothing_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let smoothing = match unsafe { smoothing_arg(smoothing_ptr, smoothing_len) } {
            Ok(value) => value,
            Err(err) => return set_error(err),
        };
        let Some(path) = (unsafe { utf8(path, path_len) }) else {
            return set_error("terrain path pointer/UTF-8 invalid");
        };
        let halo = if halo == u32::MAX { HALO } else { halo as usize };
        let built = with_grid(grid, |g| {
            crate::fields::build_terrain_smoothed(
                g,
                std::path::Path::new(path),
                halo,
                smoothing,
            )
        });
        match built {
            None => set_error(format!("unknown grid handle {grid}")),
            Some(Err(err)) => set_error(err.to_string()),
            Some(Ok(fields)) => {
                if out_handle.is_null() {
                    return set_error("out_handle is null");
                }
                unsafe { *out_handle = register_fieldset(fields) };
                OK
            }
        }
    })
}

/// Smooth one contiguous halo-extended f64 plane (`ny` rows of `nx`).
///
/// # Safety
/// `data` and `out` must each address `ny * nx` doubles; `smoothing_ptr`
/// must address `smoothing_len` bytes of UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_terrain_smooth(
    data: *const f64,
    ny: u64,
    nx: u64,
    smoothing_ptr: *const u8,
    smoothing_len: usize,
    out: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if data.is_null() || out.is_null() {
            return set_error("terrain array pointer is null");
        }
        let smoothing = match unsafe { smoothing_arg(smoothing_ptr, smoothing_len) } {
            Ok(value) => value,
            Err(err) => return set_error(err),
        };
        let (Ok(ny), Ok(nx)) = (usize::try_from(ny), usize::try_from(nx)) else {
            return set_error("terrain dimensions overflow");
        };
        let Some(n) = ny.checked_mul(nx).filter(|n| *n <= isize::MAX as usize / 8)
        else {
            return set_error("terrain dimensions overflow");
        };
        let plane = unsafe { std::slice::from_raw_parts(data, n) }.to_vec();
        let a = crate::types::Grid2 { ny, nx, data: plane };
        match crate::smooth::apply_terrain_smoothing(&a, smoothing) {
            Err(err) => set_error(err.to_string()),
            Ok(result) => {
                unsafe { std::ptr::copy_nonoverlapping(result.data.as_ptr(), out, n) };
                OK
            }
        }
    })
}
