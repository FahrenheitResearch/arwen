//! The C ABI seam `gpuwm/isobaric_bridge.py` loads through ctypes.
//!
//! Same discipline as `obs-regrid/src/capi.rs`, `static-fields/src/capi/`
//! and `netcdf-writer/src/capi.rs`: cdylib + ctypes (no pyo3 -- one loading
//! discipline across every gpuwm bridge), positional C signatures guarded
//! by an ABI version probe, a thread-local last-error string, and a
//! contract-marker symbol (`gpuwm_isobaric_heights`, listed in
//! `gpuwm/bridges.py` `BRIDGE_ABI_MARKERS`).
//!
//! Stateless: every call takes its columns and writes into caller buffers.
//! Arrays cross as raw little-endian f64, exactly `numpy.tobytes()` of a
//! C-contiguous array, level index outermost (`[k * cells + cell]`).
//!
//! The vertical coordinate crosses as the interfaces (`eta_interface`,
//! `nz + 1` values) and, when the frame states them, the mass levels
//! (`eta_mass`, `nz` values, or null for the middle of each layer in eta).
//! Interface heights cross as `interface` (plus `interface_plus`, or null:
//! WRF's PH and PHB summed here, in f64, so no caller adds them) over
//! `per_metre`.

use std::cell::RefCell;

use crate::interface::{
    InterfaceStencil, Interfaces, interface_log_pressures_into, interfaces_from_layer_thickness,
    isobaric_heights_f64_into, mass_level_heights_into,
};

/// The seam's ABI version.  Bump when a signature changes shape, never for
/// a rebuild.
pub const ISOBARIC_ABI_VERSION: u32 = 1;

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

/// Turn a panic below this seam into the ABI's own refusal: an unwind that
/// reaches an `extern "C"` boundary aborts the host interpreter.
fn guard<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            LAST_ERROR.with(|slot| {
                if let Ok(mut message) = slot.try_borrow_mut() {
                    *message = format!("panic in the isobaric-height seam: {detail}");
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

/// # Safety
/// `eta_interface` must address `nz + 1` values and `eta_mass` `nz`, or be
/// null.
unsafe fn stencil(
    eta_mass: *const f64,
    eta_interface: *const f64,
    nz: usize,
) -> Result<InterfaceStencil, String> {
    let interface = unsafe { slice(eta_interface, nz + 1) }
        .ok_or("null eta-interface pointer")?;
    if eta_mass.is_null() {
        InterfaceStencil::from_interfaces(interface)
    } else {
        let mass = unsafe { slice(eta_mass, nz) }.ok_or("null eta-mass pointer")?;
        InterfaceStencil::new(mass, interface)
    }
}

fn cells_of(nz: usize, cells: usize) -> Result<(usize, usize), String> {
    let mass = nz
        .checked_mul(cells)
        .filter(|value| *value > 0)
        .ok_or("the columns must be a non-empty shape")?;
    let interface = (nz + 1)
        .checked_mul(cells)
        .ok_or("the interface count overflows")?;
    Ok((mass, interface))
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_isobaric_abi_version() -> u32 {
    guard(0, || ISOBARIC_ABI_VERSION)
}

/// The source-revision stamp, same contract as `gpuwm_obsregrid_source_rev`:
/// read out of the binary as bytes by the release cut, never executed.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_isobaric_source_rev() -> *const std::os::raw::c_char {
    guard(std::ptr::null(), || {
        static SOURCE_REV_STAMP: &str = concat!(
            "GPUWM_BRIDGE_SOURCE_REV=",
            env!("GPUWM_BRIDGE_SOURCE_REV"),
            "\0"
        );
        SOURCE_REV_STAMP.as_ptr().cast()
    })
}

/// Copy the thread-local last error into `buf` (UTF-8, no NUL); returns the
/// full message length so a short buffer is detectable.
///
/// # Safety
/// `buf` must point to `cap` writable bytes, or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_isobaric_last_error(buf: *mut u8, cap: usize) -> usize {
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

/// The interfaces of a coordinate stated as layer thicknesses (WRF's
/// `DNW`, `nz` values) into `out_eta_interface` (`nz + 1` values).
///
/// # Safety
/// `dnw` must address `nz` values and `out_eta_interface` `nz + 1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_isobaric_interfaces_from_layer_thickness(
    dnw: *const f64,
    nz: usize,
    out_eta_interface: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (Some(dnw), Some(out)) = (unsafe { slice(dnw, nz) }, unsafe {
            slice_mut(out_eta_interface, nz + 1)
        }) else {
            return set_error("null layer-thickness or output pointer");
        };
        if nz == 0 {
            return set_error("a coordinate needs at least one layer");
        }
        out.copy_from_slice(&interfaces_from_layer_thickness(dnw));
        OK
    })
}

/// Every isobaric height (m, f64) at `levels` (same unit as `p_mass`) in
/// every column, into `out` (`n_levels * cells`, level-major), NaN where a
/// column has no such surface.  `rw_isobaric::isobaric_heights_f64_into`.
///
/// # Safety
/// Every pointer must address the number of elements the shapes imply;
/// `interface_plus` and `eta_mass` may be null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_isobaric_heights(
    interface: *const f64,
    interface_plus: *const f64,
    per_metre: f64,
    p_mass: *const f64,
    nz: usize,
    cells: usize,
    eta_mass: *const f64,
    eta_interface: *const f64,
    levels: *const f64,
    n_levels: usize,
    out: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (mass_cells, interface_cells) = match cells_of(nz, cells) {
            Ok(value) => value,
            Err(error) => return set_error(error),
        };
        let stencil = match unsafe { stencil(eta_mass, eta_interface, nz) } {
            Ok(value) => value,
            Err(error) => return set_error(error),
        };
        let (Some(values), Some(p_mass), Some(levels)) = (
            unsafe { slice(interface, interface_cells) },
            unsafe { slice(p_mass, mass_cells) },
            unsafe { slice(levels, n_levels) },
        ) else {
            return set_error("null interface, pressure or level pointer");
        };
        let interfaces = if interface_plus.is_null() {
            Interfaces::new(values, per_metre)
        } else {
            match unsafe { slice(interface_plus, interface_cells) } {
                Some(plus) => Interfaces::split(values, plus, per_metre),
                None => return set_error("null second interface pointer"),
            }
        };
        let Some(out) = (match n_levels.checked_mul(cells) {
            Some(count) => unsafe { slice_mut(out, count) },
            None => None,
        }) else {
            return set_error("null or oversized output pointer");
        };
        match isobaric_heights_f64_into(&interfaces, p_mass, &stencil, cells, levels, out) {
            Ok(()) => OK,
            Err(error) => set_error(error),
        }
    })
}

/// ln p on every interface (`(nz + 1) * cells`), NaN throughout a column
/// with an interface pressure that is not a positive number.
/// `rw_isobaric::interface_log_pressures_into`.
///
/// # Safety
/// Every pointer must address the number of elements the shapes imply;
/// `eta_mass` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_isobaric_interface_log_pressure(
    p_mass: *const f64,
    nz: usize,
    cells: usize,
    eta_mass: *const f64,
    eta_interface: *const f64,
    out: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (mass_cells, interface_cells) = match cells_of(nz, cells) {
            Ok(value) => value,
            Err(error) => return set_error(error),
        };
        let stencil = match unsafe { stencil(eta_mass, eta_interface, nz) } {
            Ok(value) => value,
            Err(error) => return set_error(error),
        };
        let (Some(p_mass), Some(out)) = (unsafe { slice(p_mass, mass_cells) }, unsafe {
            slice_mut(out, interface_cells)
        }) else {
            return set_error("null pressure or output pointer");
        };
        match interface_log_pressures_into(p_mass, &stencil, cells, out) {
            Ok(()) => OK,
            Err(error) => set_error(error),
        }
    })
}

/// The height (m) of each mass level's own pressure, `nz * cells`, read
/// between its layer's interfaces.  `rw_isobaric::mass_level_heights_into`.
///
/// # Safety
/// Every pointer must address the number of elements the shapes imply;
/// `interface_plus` and `eta_mass` may be null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_isobaric_mass_level_heights(
    interface: *const f64,
    interface_plus: *const f64,
    per_metre: f64,
    p_mass: *const f64,
    nz: usize,
    cells: usize,
    eta_mass: *const f64,
    eta_interface: *const f64,
    out: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        let (mass_cells, interface_cells) = match cells_of(nz, cells) {
            Ok(value) => value,
            Err(error) => return set_error(error),
        };
        let stencil = match unsafe { stencil(eta_mass, eta_interface, nz) } {
            Ok(value) => value,
            Err(error) => return set_error(error),
        };
        let (Some(values), Some(p_mass), Some(out)) = (
            unsafe { slice(interface, interface_cells) },
            unsafe { slice(p_mass, mass_cells) },
            unsafe { slice_mut(out, mass_cells) },
        ) else {
            return set_error("null interface, pressure or output pointer");
        };
        let interfaces = if interface_plus.is_null() {
            Interfaces::new(values, per_metre)
        } else {
            match unsafe { slice(interface_plus, interface_cells) } {
                Some(plus) => Interfaces::split(values, plus, per_metre),
                None => return set_error("null second interface pointer"),
            }
        };
        match mass_level_heights_into(&interfaces, p_mass, &stencil, cells, out) {
            Ok(()) => OK,
            Err(error) => set_error(error),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::{STANDARD_GRAVITY, ln};

    fn last_error() -> String {
        let length = unsafe { gpuwm_isobaric_last_error(std::ptr::null_mut(), 0) };
        let mut buffer = vec![0u8; length];
        unsafe { gpuwm_isobaric_last_error(buffer.as_mut_ptr(), length) };
        String::from_utf8(buffer).unwrap()
    }

    /// One isothermal eta column, PH + PHB split, read through the seam:
    /// the height is the analytic one, the mass-level heights sit on their
    /// own pressures, and the log-pressures are the interfaces'.
    #[test]
    fn the_seam_reads_an_isothermal_eta_column() {
        let nz = 30usize;
        let (surface, top, h) = (100_000.0f64, 2_000.0f64, 7_400.0f64);
        let eta: Vec<f64> = (0..=nz).map(|k| 1.0 - (k as f64 / nz as f64).powf(1.3)).collect();
        let p_w: Vec<f64> = eta.iter().map(|e| top + e * (surface - top)).collect();
        let p_m: Vec<f64> = p_w.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let phi: Vec<f64> = p_w.iter().map(|p| STANDARD_GRAVITY * h * (surface / p).ln()).collect();
        let base = vec![0.0f64; nz + 1];
        let levels = [50_000.0f64, 120_000.0];
        let mut out = [0.0f64; 2];
        let code = unsafe {
            gpuwm_isobaric_heights(
                phi.as_ptr(),
                base.as_ptr(),
                STANDARD_GRAVITY,
                p_m.as_ptr(),
                nz,
                1,
                std::ptr::null(),
                eta.as_ptr(),
                levels.as_ptr(),
                2,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, OK, "{}", last_error());
        assert!((out[0] - h * (surface / 50_000.0).ln()).abs() < 0.05);
        assert!(out[1].is_nan(), "1200 hPa is under the ground");

        let mut z_m = vec![0.0f64; nz];
        let code = unsafe {
            gpuwm_isobaric_mass_level_heights(
                phi.as_ptr(),
                std::ptr::null(),
                STANDARD_GRAVITY,
                p_m.as_ptr(),
                nz,
                1,
                std::ptr::null(),
                eta.as_ptr(),
                z_m.as_mut_ptr(),
            )
        };
        assert_eq!(code, OK, "{}", last_error());
        for (z, p) in z_m.iter().zip(&p_m) {
            assert!((z - h * (surface / p).ln()).abs() < 0.05);
        }

        let mut ln_p = vec![0.0f64; nz + 1];
        let code = unsafe {
            gpuwm_isobaric_interface_log_pressure(
                p_m.as_ptr(),
                nz,
                1,
                std::ptr::null(),
                eta.as_ptr(),
                ln_p.as_mut_ptr(),
            )
        };
        assert_eq!(code, OK, "{}", last_error());
        assert!((ln_p[0] - ln(surface)).abs() < 1e-6);

        let dnw: Vec<f64> = eta.windows(2).map(|w| w[1] - w[0]).collect();
        let mut rebuilt = vec![0.0f64; nz + 1];
        let code = unsafe {
            gpuwm_isobaric_interfaces_from_layer_thickness(dnw.as_ptr(), nz, rebuilt.as_mut_ptr())
        };
        assert_eq!(code, OK);
        assert!(rebuilt.iter().zip(&eta).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn the_seam_refuses_bad_input_by_name_and_answers_its_version() {
        assert_eq!(gpuwm_isobaric_abi_version(), ISOBARIC_ABI_VERSION);
        let eta = [1.0f64, 0.5, 0.0];
        let p = [600.0f64, 500.0];
        let z = [0.0f64, 1.0, 2.0];
        let mut out = [0.0f64; 1];
        let code = unsafe {
            gpuwm_isobaric_heights(
                z.as_ptr(),
                std::ptr::null(),
                0.0,
                p.as_ptr(),
                2,
                1,
                std::ptr::null(),
                eta.as_ptr(),
                [550.0f64].as_ptr(),
                1,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, ERR);
        assert!(last_error().contains("not a scale"), "{}", last_error());
        let code = unsafe {
            gpuwm_isobaric_heights(
                std::ptr::null(),
                std::ptr::null(),
                1.0,
                p.as_ptr(),
                2,
                1,
                std::ptr::null(),
                eta.as_ptr(),
                [550.0f64].as_ptr(),
                1,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, ERR);
        assert!(last_error().contains("null"));
        assert!(!gpuwm_isobaric_source_rev().is_null());
    }
}
