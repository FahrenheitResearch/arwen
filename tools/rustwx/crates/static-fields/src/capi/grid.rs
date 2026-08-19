//! LANE 1 seam: grid handles and coordinate/derived-field arrays.
//!
//! The Python `ProjectedGrid` classes keep their signatures; their
//! array methods route here by default.  Scalar transforms stay on the
//! Python float path only inside pure-Python fallback mode (a reported
//! workaround, never the default).

use super::{
    bytes, clear_error, drop_grid, register_grid, set_error, utf8, with_grid,
    ERR, OK,
};
use crate::corridor;
use crate::projection::{GridSpec, ProjectedGrid};
use crate::types::Stagger;

/// Create a grid from a `GridSpec` JSON document; writes the handle to
/// `out_handle`.  Nest/translated construction are separate calls so
/// the exact Python arithmetic path (parent transform, integer offsets)
/// is preserved rather than re-derived from parameters.
///
/// # Safety
/// `spec_json`/`spec_len` must describe readable UTF-8; `out_handle`
/// must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_new(
    spec_json: *const u8,
    spec_len: usize,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let Some(text) = (unsafe { utf8(spec_json, spec_len) }) else {
        return set_error("grid spec pointer/UTF-8 invalid");
    };
    let spec: GridSpec = match serde_json::from_str(text) {
        Ok(spec) => spec,
        Err(err) => return set_error(format!("grid spec JSON: {err}")),
    };
    match ProjectedGrid::new(spec) {
        Ok(grid) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            unsafe { *out_handle = register_grid(grid) };
            OK
        }
        Err(err) => set_error(err.to_string()),
    }
}

/// WPS nest arithmetic on an existing grid handle.
///
/// # Safety
/// `out_handle` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_nest(
    parent: u64,
    i_parent_start: i64,
    j_parent_start: i64,
    parent_grid_ratio: i64,
    e_we: i64,
    e_sn: i64,
    resolved_dx: f64, // NaN = None
    resolved_dy: f64, // NaN = None
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let nested = with_grid(parent, |grid| {
        grid.nest(
            i_parent_start,
            j_parent_start,
            parent_grid_ratio,
            e_we,
            e_sn,
            (!resolved_dx.is_nan()).then_some(resolved_dx),
            (!resolved_dy.is_nan()).then_some(resolved_dy),
        )
    });
    match nested {
        None => set_error(format!("unknown grid handle {parent}")),
        Some(Err(err)) => set_error(err.to_string()),
        Some(Ok(grid)) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            unsafe { *out_handle = register_grid(grid) };
            OK
        }
    }
}

/// Whole-cell placement translation (+ optional re-extent; pass <= 0 to
/// keep the reference extent).
///
/// # Safety
/// `out_handle` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_translated(
    reference: u64,
    di_cells: i64,
    dj_cells: i64,
    e_we: i64,
    e_sn: i64,
    out_handle: *mut u64,
) -> i32 {
    clear_error();
    let translated = with_grid(reference, |grid| {
        grid.translated(
            di_cells,
            dj_cells,
            (e_we > 0).then_some(e_we),
            (e_sn > 0).then_some(e_sn),
        )
    });
    match translated {
        None => set_error(format!("unknown grid handle {reference}")),
        Some(Err(err)) => set_error(err.to_string()),
        Some(Ok(grid)) => {
            if out_handle.is_null() {
                return set_error("out_handle is null");
            }
            unsafe { *out_handle = register_grid(grid) };
            OK
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_grid_free(handle: u64) {
    drop_grid(handle);
}

/// Which derived array `gpuwm_static_grid_array` fills.
/// 0 = lat, 1 = lon, 2 = mapfac, 3 = coriolis F, 4 = coriolis E,
/// 5 = sinalpha, 6 = cosalpha.
const ARRAY_KIND_MAX: u32 = 6;

/// Fill `out` (row-major f64, caller-allocated at the stagger's
/// `ny * nx`) with one derived array, delegating to
/// `ProjectedGrid::latlon` and the derived-field methods.
///
/// # Safety
/// `out` must point to `out_len` writable f64 slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_array(
    handle: u64,
    stagger_code: u32,
    array_kind: u32,
    out: *mut f64,
    out_len: usize,
) -> i32 {
    clear_error();
    if array_kind > ARRAY_KIND_MAX {
        return set_error(format!("unknown array kind {array_kind}"));
    }
    let Ok(stagger) = Stagger::from_code(stagger_code) else {
        return set_error(format!("unknown stagger code {stagger_code}"));
    };
    if out.is_null() && out_len > 0 {
        return set_error("output buffer is null");
    }
    let filled = with_grid(handle, |grid| {
        let (ny, nx, _, _) = grid.stagger_layout(stagger);
        if out_len != ny * nx {
            return Err(format!(
                "output buffer holds {out_len} f64 slots, stagger \
                 {stagger_code} needs {ny} * {nx} = {}",
                ny * nx
            ));
        }
        let data = match array_kind {
            0 => grid.latlon(stagger).0.data,
            1 => grid.latlon(stagger).1.data,
            2 => grid.map_factor_array(stagger).data,
            3 => grid.coriolis_arrays(stagger).0.data,
            4 => grid.coriolis_arrays(stagger).1.data,
            5 => grid.rotation_arrays(stagger).0.data,
            _ => grid.rotation_arrays(stagger).1.data,
        };
        let target =
            unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        target.copy_from_slice(&data);
        Ok(())
    });
    match filled {
        None => set_error(format!("unknown grid handle {handle}")),
        Some(Err(message)) => set_error(message),
        Some(Ok(())) => OK,
    }
}

/// Point transforms for many points in place; `direction` 0 =
/// ij_to_latlon (a = x -> lat, b = y -> lon), 1 = latlon_to_ij
/// (a = lat -> x, b = lon -> y).
///
/// # Safety
/// `a`/`b` must each point to `n` writable f64 slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_transform(
    handle: u64,
    direction: u32,
    a: *mut f64,
    b: *mut f64,
    n: usize,
) -> i32 {
    clear_error();
    if direction > 1 {
        return set_error(format!("unknown transform direction {direction}"));
    }
    if (a.is_null() || b.is_null()) && n > 0 {
        return set_error("transform buffers are null");
    }
    let done = with_grid(handle, |grid| {
        let av = unsafe { std::slice::from_raw_parts_mut(a, n) };
        let bv = unsafe { std::slice::from_raw_parts_mut(b, n) };
        for k in 0..n {
            let (first, second) = if direction == 0 {
                grid.ij_to_latlon(av[k], bv[k])
            } else {
                grid.latlon_to_ij(av[k], bv[k])
            };
            av[k] = first;
            bv[k] = second;
        }
    });
    match done {
        None => set_error(format!("unknown grid handle {handle}")),
        Some(()) => OK,
    }
}

/// Corridor identity probes as JSON (`grid_identity_probes`): the five
/// probe points in the receipt's own order, float values formatted
/// shortest-round-trip (they parse back to the exact float64 bits the
/// Python receipts carry).  Copies up to `cap` bytes into `buf` and
/// returns the full rendered length (call again with a larger buffer if
/// it was truncated); -1 on error.
///
/// # Safety
/// `buf` must point to `cap` writable bytes or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_identity_probes(
    handle: u64,
    buf: *mut u8,
    cap: usize,
) -> i64 {
    clear_error();
    let rendered = with_grid(handle, |grid| {
        corridor::grid_identity_probes(grid).map(|probes| {
            let mut out = String::from("{");
            for (index, (name, [lat, lon])) in probes.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "\"{name}\":[{},{}]",
                    serde_json::to_string(lat).unwrap_or_default(),
                    serde_json::to_string(lon).unwrap_or_default(),
                ));
            }
            out.push('}');
            out
        })
    });
    match rendered {
        None => {
            set_error(format!("unknown grid handle {handle}"));
            ERR as i64
        }
        Some(Err(err)) => {
            set_error(err.to_string());
            ERR as i64
        }
        Some(Ok(json)) => {
            let raw = json.as_bytes();
            if !buf.is_null() && cap > 0 {
                let n = raw.len().min(cap);
                unsafe {
                    std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n);
                }
            }
            raw.len() as i64
        }
    }
}

/// Keep `bytes` referenced for the lanes that will need raw-byte config
/// crossings on this file's calls (silences no-op warnings without
/// removing the helper from the seam surface).
#[allow(dead_code)]
fn _reserved(ptr: *const u8, len: usize) -> usize {
    unsafe { bytes(ptr, len) }.map_or(0, <[u8]>::len)
}
