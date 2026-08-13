//! C-ABI surface over [`solver`], built for `wasm32-unknown-unknown` (and
//! usable as a plain shared library).
//!
//! Everything crosses the boundary as flat arrays of `f32` in linear memory, so
//! a host needs no `wasm-bindgen` glue and no generated JavaScript:
//!
//! ```text
//! bw_abi_version() -> u32
//! bw_alloc(len) -> ptr            // 16-byte aligned, or null
//! bw_free(ptr, len)
//! bw_dealias(vel, azim, nyq, rows, gates, out, stats) -> i32
//! ```
//!
//! `vel` and `out` are `rows * gates` floats in **row-major ray order** (ray
//! 0's gates first). `azim` and `nyq` are `rows` floats. A non-finite velocity
//! means "no data" and is passed through untouched; a non-finite or
//! non-positive Nyquist means that ray's Nyquist is unknown, and the solver
//! falls back to the sweep median.
//!
//! Return codes are [`BW_OK`] and the negative `BW_ERR_*` constants.

use std::alloc::Layout;

pub mod solver;

/// Bumped on any breaking change to the exported function signatures or to the
/// [`BwStats`] layout. Hosts should refuse a module whose value they don't know.
pub const BW_ABI_VERSION: u32 = 1;
/// Version of the additive RIFT-VDA entry points. This is intentionally
/// independent from [`BW_ABI_VERSION`], whose legacy surface remains frozen.
pub const BW_RIFT_API_VERSION: u32 = 1;

/// Alignment every [`bw_alloc`] block is handed out at. Comfortably above the
/// 4 bytes an `f32` view needs, so a host can safely place `Float32Array`,
/// `Uint32Array`, or `Float64Array` views on any returned pointer.
const BW_ALIGN: usize = 16;

/// Upper bound on `rows * gates`.
///
/// A NEXRAD super-resolution velocity sweep is ~0.86 M gates (720 x 1192), and
/// the widest realistic sweep of any kind is ~1.3 M. This allows ~4.9x the
/// former, so no plausible radar data is rejected, while bounding the damage a
/// corrupt length can do: solver memory scales with the number of *regions*,
/// which on pathological (non-radar) input approaches one per gate at roughly
/// 185 bytes each. A larger ceiling would permit a multi-gigabyte allocation —
/// on wasm32, an unrecoverable abort rather than an error return.
const BW_MAX_GATES: usize = 1 << 22;

/// Success.
pub const BW_OK: i32 = 0;
/// A required pointer was null.
pub const BW_ERR_NULL: i32 = -1;
/// `rows`/`gates` were zero, overflowed, or exceeded the gate ceiling.
pub const BW_ERR_DIMS: i32 = -2;
/// A pointer was not 4-byte aligned, so it cannot be read as `f32`.
pub const BW_ERR_ALIGN: i32 = -3;
/// A versioned options structure or flag set is not understood.
pub const BW_ERR_VERSION: i32 = -4;
/// An advanced input has a length inconsistent with the sweep.
pub const BW_ERR_LENGTH: i32 = -5;
/// A configured RIFT safety budget is outside its supported range.
pub const BW_ERR_LIMIT: i32 = -6;
/// A caller-supplied reference field or kind is invalid.
pub const BW_ERR_REFERENCE: i32 = -7;
/// Input and output memory ranges overlap, or a range overflows the address space.
pub const BW_ERR_ALIAS: i32 = -8;
/// Required polar range geometry is missing or non-physical.
pub const BW_ERR_GEOMETRY: i32 = -9;

/// Per-call diagnostics, written through the optional `stats` out-pointer.
///
/// `#[repr(C)]` with five `u32`s: 20 bytes, no padding, so a host can read it
/// as a `Uint32Array` of length 5.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BwStats {
    /// `rows * gates`.
    pub gates_total: u32,
    /// Gates that carried a finite velocity on input.
    pub gates_finite: u32,
    /// Gates the solver moved by at least one Nyquist interval.
    pub gates_modified: u32,
    /// Largest `|fold|` applied anywhere on the sweep.
    pub max_abs_fold: u32,
    /// `1` when the ray order closed a full sweep, so the seam between the last
    /// and first ray was available as an adjacency; `0` for a partial sector,
    /// which is solved without that edge. A caller streaming sector chunks can
    /// watch this to know whether it is handing over whole sweeps.
    pub wraps: u32,
}

/// Caller configuration for [`bw_dealias_rift_v1`]. `struct_size` must be at
/// least `sizeof(BwRiftOptionsV1)` and may contain only known `BW_RIFT_FLAG_*`
/// bits. A null options pointer selects safe reference-only defaults.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BwRiftOptionsV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub max_abs_fold: u32,
    pub max_rois: u32,
    pub max_roi_gates: u32,
    pub max_total_roi_gates: u32,
    pub min_confidence: u32,
    /// Slant range to gate zero in metres. Must be finite and non-negative when
    /// automatic single-sweep proposals are enabled.
    pub first_gate_m: f32,
    /// Gate-center spacing in metres. Must be finite and positive when
    /// automatic single-sweep proposals are enabled.
    pub gate_spacing_m: f32,
    pub reserved: [u32; 3],
}

impl Default for BwRiftOptionsV1 {
    fn default() -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            flags: BW_RIFT_FLAG_DISABLE_AUTOMATIC_SINGLE_SWEEP,
            max_abs_fold: 4,
            max_rois: 4,
            max_roi_gates: 65_536,
            max_total_roi_gates: 0,
            min_confidence: 160,
            first_gate_m: f32::NAN,
            gate_spacing_m: f32::NAN,
            reserved: [0; 3],
        }
    }
}

/// Diagnostics for [`bw_dealias_rift_v1`]. The versioned structure is 64 bytes
/// and consists only of `u32`, making it straightforward to read from WASM.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BwRiftStatsV1 {
    pub struct_size: u32,
    pub gates_total: u32,
    pub gates_finite: u32,
    pub gates_modified: u32,
    pub max_abs_fold: u32,
    pub wraps: u32,
    pub rois_detected: u32,
    pub rois_solved: u32,
    pub rois_accepted: u32,
    pub gates_refined: u32,
    pub gates_ambiguous: u32,
    pub budget_aborts: u32,
    pub reason_flags: u32,
    pub abstain_flags: u32,
    pub reserved: [u32; 2],
}

pub const BW_RIFT_REF_CALLER: u8 = 0;
pub const BW_RIFT_REF_TEMPORAL: u8 = 1;
pub const BW_RIFT_REF_VERTICAL: u8 = 2;
pub const BW_RIFT_REF_ENVIRONMENTAL: u8 = 3;
/// Disable velocity-only single-sweep proposals. This is appropriate for
/// reference-only callers and makes range geometry optional.
pub const BW_RIFT_FLAG_DISABLE_AUTOMATIC_SINGLE_SWEEP: u32 = 1 << 0;
const BW_RIFT_KNOWN_FLAGS: u32 = BW_RIFT_FLAG_DISABLE_AUTOMATIC_SINGLE_SWEEP;

/// The ABI version this module was built with.
#[unsafe(no_mangle)]
pub extern "C" fn bw_abi_version() -> u32 {
    BW_ABI_VERSION
}

/// The additive RIFT-VDA API version supported by this module.
#[unsafe(no_mangle)]
pub extern "C" fn bw_rift_api_version() -> u32 {
    BW_RIFT_API_VERSION
}

/// Allocate `len` bytes, aligned to [`BW_ALIGN`]. Returns null for `len == 0`
/// or on allocation failure.
///
/// The caller owns the block and must return it to [`bw_free`] with the same
/// `len`.
#[unsafe(no_mangle)]
pub extern "C" fn bw_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    match Layout::from_size_align(len, BW_ALIGN) {
        // SAFETY: `len` is non-zero, so the layout has non-zero size.
        Ok(layout) => unsafe { std::alloc::alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Release a block from [`bw_alloc`].
///
/// # Safety
/// `ptr` must be null, or a pointer returned by [`bw_alloc`] that has not been
/// freed, and `len` must be the exact length it was allocated with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bw_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, BW_ALIGN) {
        // SAFETY: the caller guarantees `ptr` came from `bw_alloc` with this
        // same `len`, and `BW_ALIGN` is the alignment it was allocated at.
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

/// Dealias one sweep.
///
/// Writes `rows * gates` floats to `out` (non-finite where the input had no
/// data) and, when `stats` is non-null, a [`BwStats`] describing the solve.
///
/// Returns [`BW_OK`], or a negative `BW_ERR_*` code, in which case nothing is
/// written.
///
/// # Safety
/// - `vel` must point to `rows * gates` readable, 4-byte-aligned `f32`s.
/// - `azim` and `nyq` must each point to `rows` readable, aligned `f32`s.
/// - `out` must point to `rows * gates` writable, aligned `f32`s, and must not
///   alias `vel`.
/// - `stats`, if non-null, must point to a writable, 4-byte-aligned [`BwStats`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bw_dealias(
    vel: *const f32,
    azim: *const f32,
    nyq: *const f32,
    rows: usize,
    gates: usize,
    out: *mut f32,
    stats: *mut BwStats,
) -> i32 {
    if vel.is_null() || azim.is_null() || nyq.is_null() || out.is_null() {
        return BW_ERR_NULL;
    }
    if !aligned4(vel) || !aligned4(azim) || !aligned4(nyq) || !aligned4(out) || !aligned4(stats) {
        return BW_ERR_ALIGN;
    }
    let Some(total) = checked_total(rows, gates) else {
        return BW_ERR_DIMS;
    };

    // SAFETY: the caller guarantees these lengths and alignments; both were
    // checked above, and `total` is `rows * gates` without overflow.
    let (vel, azim, nyq) = unsafe {
        (
            std::slice::from_raw_parts(vel, total),
            std::slice::from_raw_parts(azim, rows),
            std::slice::from_raw_parts(nyq, rows),
        )
    };

    // `solver` normalizes nothing about azimuths, so match the convention the
    // seam test expects before handing them over.
    let normalized: Vec<f32> = azim.iter().map(|a| a.rem_euclid(360.0)).collect();
    let solved = solver::dealias_sweep(vel, nyq, rows, gates, &normalized);

    // SAFETY: `out` is valid for `total` writable, aligned floats per the
    // contract, and does not alias `vel`.
    let out = unsafe { std::slice::from_raw_parts_mut(out, total) };
    out.copy_from_slice(&solved);

    if !stats.is_null() {
        let computed = summarize(vel, out, nyq, rows, gates, &normalized);
        // SAFETY: non-null and aligned (checked above), and `BwStats` is a
        // plain `#[repr(C)]` POD.
        unsafe { stats.write(computed) };
    }
    BW_OK
}

/// Advanced, opt-in RIFT-VDA single-sweep solve.
///
/// References use flat structure-of-arrays storage: `reference_velocity` is
/// `reference_count * rows * gates` `f32`s, `reference_quality` is either null
/// or the same number of `u8`s, and `reference_kind` is one `u8` per reference.
/// Each reference is already projected onto the current sweep's grid. A
/// non-finite reference velocity is absent; absent quality means 255 wherever
/// its velocity is finite.
///
/// `reflectivity`, `spectrum_width`, and `rho_hv` are optional `rows * gates`
/// confidence fields. They do not determine truth and may be null. The three
/// diagnostic outputs and `stats` may also be null independently. The primary
/// `out_velocity` is required.
///
/// # Safety
/// Every non-null pointer must be valid for the element count documented above
/// and aligned for its element type. Writable output ranges must not overlap
/// any input range or one another.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bw_dealias_rift_v1(
    vel: *const f32,
    azim: *const f32,
    nyq: *const f32,
    rows: usize,
    gates: usize,
    reference_velocity: *const f32,
    reference_quality: *const u8,
    reference_kind: *const u8,
    reference_count: u32,
    reflectivity: *const f32,
    spectrum_width: *const f32,
    rho_hv: *const f32,
    options: *const BwRiftOptionsV1,
    out_velocity: *mut f32,
    out_folds: *mut i8,
    out_confidence: *mut u8,
    out_reasons: *mut u16,
    stats: *mut BwRiftStatsV1,
) -> i32 {
    if vel.is_null() || azim.is_null() || nyq.is_null() || out_velocity.is_null() {
        return BW_ERR_NULL;
    }
    let Some(total) = checked_total(rows, gates) else {
        return BW_ERR_DIMS;
    };
    let reference_count = reference_count as usize;
    if reference_count > 4 {
        return BW_ERR_LIMIT;
    }
    if reference_count > 0 && (reference_velocity.is_null() || reference_kind.is_null()) {
        return BW_ERR_NULL;
    }
    let Some(reference_total) = total.checked_mul(reference_count) else {
        return BW_ERR_DIMS;
    };
    if !aligned4(vel)
        || !aligned4(azim)
        || !aligned4(nyq)
        || !aligned4(reference_velocity)
        || !aligned4(reflectivity)
        || !aligned4(spectrum_width)
        || !aligned4(rho_hv)
        || !aligned4(options)
        || !aligned4(out_velocity)
        || !aligned_to(out_reasons, 2)
        || !aligned4(stats)
    {
        return BW_ERR_ALIGN;
    }

    let mut options_bytes = 0usize;
    let solver_options = if options.is_null() {
        solver::RiftOptions {
            automatic_single_sweep: false,
            ..solver::RiftOptions::default()
        }
    } else {
        // SAFETY: every supported version starts with an aligned `u32` size;
        // read only that field before deciding the known v1 prefix is present.
        let struct_size = unsafe { options.cast::<u32>().read() };
        if struct_size < size_u32::<BwRiftOptionsV1>() {
            return BW_ERR_VERSION;
        }
        options_bytes = struct_size as usize;
        // SAFETY: the size check above establishes that the caller promises at
        // least the complete readable v1 prefix.
        let value = unsafe { options.read() };
        if value.flags & !BW_RIFT_KNOWN_FLAGS != 0 {
            return BW_ERR_VERSION;
        }
        let (Ok(max_abs_fold), Ok(max_rois), Ok(min_confidence)) = (
            u8::try_from(value.max_abs_fold),
            u8::try_from(value.max_rois),
            u8::try_from(value.min_confidence),
        ) else {
            return BW_ERR_LIMIT;
        };
        let automatic_single_sweep = value.flags & BW_RIFT_FLAG_DISABLE_AUTOMATIC_SINGLE_SWEEP == 0;
        if automatic_single_sweep
            && (!value.first_gate_m.is_finite()
                || value.first_gate_m < 0.0
                || !value.gate_spacing_m.is_finite()
                || value.gate_spacing_m <= 0.0)
        {
            return BW_ERR_GEOMETRY;
        }
        solver::RiftOptions {
            max_abs_fold,
            max_rois,
            max_roi_gates: value.max_roi_gates,
            max_total_roi_gates: value.max_total_roi_gates,
            min_confidence,
            first_gate_m: value.first_gate_m,
            gate_spacing_m: value.gate_spacing_m,
            automatic_single_sweep,
        }
    };

    let input_ranges = [
        byte_range(vel, bytes_for::<f32>(total)),
        byte_range(azim, bytes_for::<f32>(rows)),
        byte_range(nyq, bytes_for::<f32>(rows)),
        byte_range(reference_velocity, bytes_for::<f32>(reference_total)),
        byte_range(reference_quality, reference_total),
        byte_range(reference_kind, reference_count),
        byte_range(reflectivity, bytes_for::<f32>(total)),
        byte_range(spectrum_width, bytes_for::<f32>(total)),
        byte_range(rho_hv, bytes_for::<f32>(total)),
        byte_range(options, options_bytes),
    ];
    let output_ranges = [
        byte_range(out_velocity, bytes_for::<f32>(total)),
        byte_range(out_folds, total),
        byte_range(out_confidence, total),
        byte_range(out_reasons, bytes_for::<u16>(total)),
        byte_range(stats, std::mem::size_of::<BwRiftStatsV1>()),
    ];
    if ranges_invalid_or_overlap(&input_ranges, &output_ranges) {
        return BW_ERR_ALIAS;
    }

    // SAFETY: shape, required pointers, and alignments were checked above; the
    // remaining validity/readability requirement belongs to the unsafe caller.
    let (vel, azim, nyq) = unsafe {
        (
            std::slice::from_raw_parts(vel, total),
            std::slice::from_raw_parts(azim, rows),
            std::slice::from_raw_parts(nyq, rows),
        )
    };
    let normalized: Vec<f32> = azim.iter().map(|a| a.rem_euclid(360.0)).collect();

    let flat_reference_velocity = if reference_count == 0 {
        &[][..]
    } else {
        // SAFETY: covered by the function contract and checked count.
        unsafe { std::slice::from_raw_parts(reference_velocity, reference_total) }
    };
    let flat_reference_quality = if reference_quality.is_null() {
        None
    } else {
        // SAFETY: covered by the function contract and checked count.
        Some(unsafe { std::slice::from_raw_parts(reference_quality, reference_total) })
    };
    let kinds = if reference_count == 0 {
        &[][..]
    } else {
        // SAFETY: covered by the function contract and checked count.
        unsafe { std::slice::from_raw_parts(reference_kind, reference_count) }
    };
    let mut references = Vec::with_capacity(reference_count);
    for (reference_index, &kind) in kinds.iter().enumerate() {
        let kind = match kind {
            BW_RIFT_REF_CALLER => solver::ReferenceKind::Caller,
            BW_RIFT_REF_TEMPORAL => solver::ReferenceKind::Temporal,
            BW_RIFT_REF_VERTICAL => solver::ReferenceKind::Vertical,
            BW_RIFT_REF_ENVIRONMENTAL => solver::ReferenceKind::Environmental,
            _ => return BW_ERR_REFERENCE,
        };
        let start = reference_index * total;
        let end = start + total;
        references.push(solver::ReferenceField {
            velocity: &flat_reference_velocity[start..end],
            quality: flat_reference_quality.map(|quality| &quality[start..end]),
            kind,
        });
    }

    // SAFETY: optional field pointers are either null or valid for `total`
    // entries by the function contract; alignment was checked above.
    let optional_field = |pointer: *const f32| {
        (!pointer.is_null()).then(|| unsafe { std::slice::from_raw_parts(pointer, total) })
    };
    let context = solver::RiftContext {
        references: &references,
        reflectivity: optional_field(reflectivity),
        spectrum_width: optional_field(spectrum_width),
        rho_hv: optional_field(rho_hv),
    };
    let result = match solver::dealias_sweep_rift(
        vel,
        nyq,
        rows,
        gates,
        &normalized,
        &context,
        solver_options,
    ) {
        Ok(result) => result,
        Err(error) => return rift_error_code(error),
    };

    // No output is touched until every validation and the solve have succeeded.
    // SAFETY: all output pointers are non-overlapping and valid by contract.
    let out = unsafe { std::slice::from_raw_parts_mut(out_velocity, total) };
    out.copy_from_slice(&result.velocity);
    if !out_folds.is_null() {
        unsafe { std::slice::from_raw_parts_mut(out_folds, total) }.copy_from_slice(&result.folds);
    }
    if !out_confidence.is_null() {
        unsafe { std::slice::from_raw_parts_mut(out_confidence, total) }
            .copy_from_slice(&result.confidence);
    }
    if !out_reasons.is_null() {
        unsafe { std::slice::from_raw_parts_mut(out_reasons, total) }
            .copy_from_slice(&result.reasons);
    }

    if !stats.is_null() {
        let base = summarize(vel, out, nyq, rows, gates, &normalized);
        let reason_flags = result
            .reasons
            .iter()
            .fold(0u16, |all, &reason| all | reason);
        let abstain_mask = solver::RIFT_REASON_CONFLICTING_REFERENCES
            | solver::RIFT_REASON_LOW_COVERAGE
            | solver::RIFT_REASON_ABSTAINED
            | solver::RIFT_REASON_BUDGET_EXCEEDED
            | solver::RIFT_REASON_NYQUIST_TRANSITION;
        let computed = BwRiftStatsV1 {
            struct_size: size_u32::<BwRiftStatsV1>(),
            gates_total: base.gates_total,
            gates_finite: base.gates_finite,
            gates_modified: base.gates_modified,
            max_abs_fold: base.max_abs_fold,
            wraps: base.wraps,
            rois_detected: result.stats.rois_detected,
            rois_solved: result.stats.rois_solved,
            rois_accepted: result.stats.rois_accepted,
            gates_refined: result.stats.gates_refined,
            gates_ambiguous: result.stats.gates_ambiguous,
            budget_aborts: result.stats.budget_aborts,
            reason_flags: reason_flags as u32,
            abstain_flags: (reason_flags & abstain_mask) as u32,
            reserved: [0; 2],
        };
        unsafe { stats.write(computed) };
    }
    BW_OK
}

fn aligned4<T>(ptr: *const T) -> bool {
    ptr.is_null() || (ptr as usize).is_multiple_of(4)
}

fn aligned_to<T>(ptr: *const T, alignment: usize) -> bool {
    ptr.is_null() || (ptr as usize).is_multiple_of(alignment)
}

fn bytes_for<T>(elements: usize) -> usize {
    elements.saturating_mul(std::mem::size_of::<T>())
}

fn size_u32<T>() -> u32 {
    u32::try_from(std::mem::size_of::<T>()).expect("ABI structure size fits u32")
}

#[derive(Clone, Copy)]
struct ByteRange {
    start: usize,
    end: usize,
    valid: bool,
}

fn byte_range<T>(pointer: *const T, bytes: usize) -> Option<ByteRange> {
    if pointer.is_null() || bytes == 0 {
        return None;
    }
    let start = pointer as usize;
    Some(match start.checked_add(bytes) {
        Some(end) => ByteRange {
            start,
            end,
            valid: true,
        },
        None => ByteRange {
            start,
            end: start,
            valid: false,
        },
    })
}

fn ranges_overlap(first: ByteRange, second: ByteRange) -> bool {
    first.start < second.end && second.start < first.end
}

fn ranges_invalid_or_overlap<const I: usize, const O: usize>(
    inputs: &[Option<ByteRange>; I],
    outputs: &[Option<ByteRange>; O],
) -> bool {
    if inputs
        .iter()
        .chain(outputs.iter())
        .flatten()
        .any(|range| !range.valid)
    {
        return true;
    }
    for output in outputs.iter().flatten().copied() {
        if inputs
            .iter()
            .flatten()
            .copied()
            .any(|input| ranges_overlap(input, output))
        {
            return true;
        }
    }
    for first in 0..outputs.len() {
        let Some(first_range) = outputs[first] else {
            continue;
        };
        if outputs[first + 1..]
            .iter()
            .flatten()
            .copied()
            .any(|second| ranges_overlap(first_range, second))
        {
            return true;
        }
    }
    false
}

fn rift_error_code(error: solver::RiftError) -> i32 {
    match error {
        solver::RiftError::Dimensions => BW_ERR_DIMS,
        solver::RiftError::Length => BW_ERR_LENGTH,
        solver::RiftError::Limit => BW_ERR_LIMIT,
        solver::RiftError::Reference => BW_ERR_REFERENCE,
    }
}

fn checked_total(rows: usize, gates: usize) -> Option<usize> {
    let total = rows.checked_mul(gates)?;
    (total > 0 && total <= BW_MAX_GATES).then_some(total)
}

fn summarize(
    vel: &[f32],
    out: &[f32],
    nyq: &[f32],
    rows: usize,
    gates: usize,
    normalized_azimuths: &[f32],
) -> BwStats {
    let resolved = solver::resolve_nyquist(nyq, rows);
    let mut stats = BwStats {
        gates_total: u32::try_from(vel.len()).unwrap_or(u32::MAX),
        wraps: u32::from(solver::sweep_wraps(normalized_azimuths)),
        ..BwStats::default()
    };

    for (row, &n) in resolved.iter().enumerate().take(rows) {
        let interval = if n.is_finite() && n > 0.0 {
            2.0 * n
        } else {
            f32::NAN
        };
        for gate in 0..gates {
            let index = row * gates + gate;
            let (before, after) = (vel[index], out[index]);
            if !before.is_finite() {
                continue;
            }
            stats.gates_finite += 1;
            if !after.is_finite() || !interval.is_finite() {
                continue;
            }
            let fold = ((after - before) / interval).round().abs();
            if fold >= 1.0 {
                stats.gates_modified += 1;
                stats.max_abs_fold = stats.max_abs_fold.max(fold as u32);
            }
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    const NYQUIST: f32 = 20.0;

    /// Wrap a true velocity into the +/- Nyquist window the radar would report.
    fn fold(value: f32, nyquist: f32) -> f32 {
        let interval = 2.0 * nyquist;
        (value + nyquist).rem_euclid(interval) - nyquist
    }

    /// A sweep whose true velocity ramps well past Nyquist along each ray, so
    /// the observed field necessarily contains fold discontinuities.
    fn folded_ramp(rows: usize, gates: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut vel = vec![f32::NAN; rows * gates];
        for row in 0..rows {
            for gate in 0..gates {
                let truth = -55.0 + 110.0 * (gate as f32 / gates as f32);
                vel[row * gates + gate] = fold(truth, NYQUIST);
            }
        }
        let azim = (0..rows)
            .map(|row| row as f32 * 360.0 / rows as f32)
            .collect();
        let nyq = vec![NYQUIST; rows];
        (vel, azim, nyq)
    }

    fn run(
        vel: &[f32],
        azim: &[f32],
        nyq: &[f32],
        rows: usize,
        gates: usize,
    ) -> (i32, Vec<f32>, BwStats) {
        let mut out = vec![0.0f32; rows * gates];
        let mut stats = BwStats::default();
        let code = unsafe {
            bw_dealias(
                vel.as_ptr(),
                azim.as_ptr(),
                nyq.as_ptr(),
                rows,
                gates,
                out.as_mut_ptr(),
                &mut stats,
            )
        };
        (code, out, stats)
    }

    /// The property that matters: after unfolding, no two adjacent gates along
    /// a ray jump by more than a Nyquist interval. Returns the largest jump.
    fn max_adjacent_jump(field: &[f32], rows: usize, gates: usize) -> f32 {
        let mut worst = 0.0f32;
        for row in 0..rows {
            for gate in 1..gates {
                let a = field[row * gates + gate - 1];
                let b = field[row * gates + gate];
                if a.is_finite() && b.is_finite() {
                    worst = worst.max((a - b).abs());
                }
            }
        }
        worst
    }

    #[test]
    fn removes_fold_discontinuities() {
        let (rows, gates) = (36, 120);
        let (vel, azim, nyq) = folded_ramp(rows, gates);

        // The test is only meaningful if the input really is folded.
        let raw_jump = max_adjacent_jump(&vel, rows, gates);
        assert!(
            raw_jump > NYQUIST,
            "fixture is not aliased (max jump {raw_jump})"
        );

        let (code, out, stats) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(code, BW_OK);
        let jump = max_adjacent_jump(&out, rows, gates);
        assert!(jump < NYQUIST, "left a {jump} m/s jump");
        assert_eq!(stats.gates_total, (rows * gates) as u32);
        assert_eq!(stats.gates_finite, (rows * gates) as u32);
        assert!(stats.gates_modified > 0, "solver changed nothing");
        assert!(stats.max_abs_fold >= 1);
        assert_eq!(stats.wraps, 1, "a full sweep should report a closed seam");
    }

    #[test]
    fn is_deterministic() {
        let (rows, gates) = (36, 120);
        let (vel, azim, nyq) = folded_ramp(rows, gates);
        let (_, first, first_stats) = run(&vel, &azim, &nyq, rows, gates);
        let (_, second, second_stats) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "output is not byte-identical across runs"
        );
        assert_eq!(first_stats, second_stats);
    }

    /// A partial sector must still solve, and must say the seam was missing.
    #[test]
    fn partial_sector_solves_and_reports_open_seam() {
        let (rows, gates) = (36, 120);
        let (vel, _, nyq) = folded_ramp(rows, gates);
        // 2 deg spacing over a 70 deg sector — the ray density a real sector
        // feed has, rather than a handful of rays smeared across the arc.
        let azim: Vec<f32> = (0..rows).map(|row| 100.0 + row as f32 * 2.0).collect();
        let (code, out, stats) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(code, BW_OK);
        assert_eq!(stats.wraps, 0);
        assert!(max_adjacent_jump(&out, rows, gates) < NYQUIST);
    }

    /// Pins the assumption behind [`BwStats::wraps`]: ray spacing is estimated
    /// as `360 / rows`, so a sector whose rays are packed much tighter than
    /// that estimate reads as closed. Real sector feeds keep the radar's native
    /// ray spacing and are unaffected.
    #[test]
    fn wraps_heuristic_assumes_rays_span_the_full_circle() {
        let realistic_sector: Vec<f32> = (0..180).map(|r| r as f32 * 0.5).collect();
        assert!(
            !solver::sweep_wraps(&realistic_sector),
            "a 90 deg super-res sector must read as open"
        );

        let full_sweep: Vec<f32> = (0..36).map(|r| r as f32 * 10.0).collect();
        assert!(solver::sweep_wraps(&full_sweep));

        // Documented limitation, asserted so a future change surfaces here.
        let sparse_sector: Vec<f32> = (0..36).map(|r| 100.0 + r as f32 * 0.5).collect();
        assert!(solver::sweep_wraps(&sparse_sector));
    }

    #[test]
    fn no_data_gates_pass_through_as_non_finite() {
        let (rows, gates) = (36, 120);
        let (mut vel, azim, nyq) = folded_ramp(rows, gates);
        for slot in vel.iter_mut().take(gates) {
            *slot = f32::NAN;
        }
        let (code, out, stats) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(code, BW_OK);
        assert!(out[..gates].iter().all(|v| !v.is_finite()));
        assert_eq!(stats.gates_finite, ((rows - 1) * gates) as u32);
    }

    /// An all-no-data sweep is a real live-feed case (a tilt with no returns).
    #[test]
    fn all_no_data_sweep_is_not_an_error() {
        let (rows, gates) = (36, 120);
        let vel = vec![f32::NAN; rows * gates];
        let azim: Vec<f32> = (0..rows).map(|r| r as f32 * 10.0).collect();
        let nyq = vec![NYQUIST; rows];
        let (code, out, stats) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(code, BW_OK);
        assert!(out.iter().all(|v| !v.is_finite()));
        assert_eq!(stats.gates_finite, 0);
        assert_eq!(stats.gates_modified, 0);
    }

    /// Unknown Nyquist must degrade to an exact pass-through, never a panic.
    #[test]
    fn unusable_nyquist_passes_velocities_through_exactly() {
        let (rows, gates) = (36, 120);
        let (vel, azim, _) = folded_ramp(rows, gates);
        for bad in [f32::NAN, 0.0, -5.0, f32::INFINITY] {
            let nyq = vec![bad; rows];
            let (code, out, stats) = run(&vel, &azim, &nyq, rows, gates);
            assert_eq!(code, BW_OK, "nyquist {bad}");
            assert_eq!(stats.gates_modified, 0);
            assert_eq!(
                vel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "nyquist {bad} should be an exact pass-through"
            );
        }
    }

    #[test]
    fn extreme_and_non_finite_velocities_do_not_panic() {
        let (rows, gates) = (16, 64);
        let mut vel = vec![0.0f32; rows * gates];
        for (index, slot) in vel.iter_mut().enumerate() {
            *slot = match index % 5 {
                0 => 1e30,
                1 => -1e30,
                2 => f32::INFINITY,
                3 => f32::NAN,
                _ => 3.0,
            };
        }
        let azim: Vec<f32> = (0..rows).map(|r| r as f32 * 22.5).collect();
        let nyq = vec![NYQUIST; rows];
        let (code, _, _) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(code, BW_OK);
    }

    /// Fewer than 8 rays is below the seam test's floor, and single-ray or
    /// single-gate sweeps are degenerate shapes a bad feed can produce.
    #[test]
    fn degenerate_shapes_solve_without_panicking() {
        for (rows, gates) in [(1usize, 64usize), (64, 1), (3, 3), (1, 1)] {
            let vel = vec![5.0f32; rows * gates];
            let azim: Vec<f32> = (0..rows).map(|r| r as f32).collect();
            let nyq = vec![NYQUIST; rows];
            let (code, _, stats) = run(&vel, &azim, &nyq, rows, gates);
            assert_eq!(code, BW_OK, "{rows}x{gates}");
            assert_eq!(stats.wraps, 0);
        }
    }

    #[test]
    fn non_finite_azimuths_are_tolerated() {
        let (rows, gates) = (36, 120);
        let (vel, _, nyq) = folded_ramp(rows, gates);
        let azim = vec![f32::NAN; rows];
        let (code, _, stats) = run(&vel, &azim, &nyq, rows, gates);
        assert_eq!(code, BW_OK);
        assert_eq!(stats.wraps, 0);
    }

    /// Azimuths outside 0..360 must behave exactly like their normalized form.
    #[test]
    fn azimuths_are_normalized() {
        let (rows, gates) = (36, 120);
        let (vel, azim, nyq) = folded_ramp(rows, gates);
        let shifted: Vec<f32> = azim.iter().map(|a| a + 720.0).collect();
        let (_, plain, _) = run(&vel, &azim, &nyq, rows, gates);
        let (_, wrapped, _) = run(&vel, &shifted, &nyq, rows, gates);
        assert_eq!(
            plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            wrapped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn rejects_bad_dimensions() {
        let vel = [0.0f32; 8];
        let azim = [0.0f32; 8];
        let nyq = [NYQUIST; 8];
        let mut out = [0.0f32; 8];
        for (rows, gates) in [(0usize, 8usize), (8, 0), (0, 0), (usize::MAX, 2)] {
            let code = unsafe {
                bw_dealias(
                    vel.as_ptr(),
                    azim.as_ptr(),
                    nyq.as_ptr(),
                    rows,
                    gates,
                    out.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(code, BW_ERR_DIMS, "{rows}x{gates}");
        }
    }

    #[test]
    fn rejects_oversized_grids() {
        let vel = [0.0f32; 8];
        let azim = [0.0f32; 8];
        let nyq = [NYQUIST; 8];
        let mut out = [0.0f32; 8];
        let code = unsafe {
            bw_dealias(
                vel.as_ptr(),
                azim.as_ptr(),
                nyq.as_ptr(),
                BW_MAX_GATES,
                2,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, BW_ERR_DIMS);
    }

    #[test]
    fn rejects_null_pointers() {
        let azim = [0.0f32; 4];
        let nyq = [NYQUIST; 4];
        let mut out = [0.0f32; 4];
        let code = unsafe {
            bw_dealias(
                std::ptr::null(),
                azim.as_ptr(),
                nyq.as_ptr(),
                1,
                4,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, BW_ERR_NULL);
    }

    #[test]
    fn stats_pointer_is_optional() {
        let (rows, gates) = (16, 64);
        let (vel, azim, nyq) = folded_ramp(rows, gates);
        let mut out = vec![0.0f32; rows * gates];
        let code = unsafe {
            bw_dealias(
                vel.as_ptr(),
                azim.as_ptr(),
                nyq.as_ptr(),
                rows,
                gates,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, BW_OK);
    }

    #[test]
    fn alloc_round_trips_and_is_aligned() {
        let ptr = bw_alloc(1024);
        assert!(!ptr.is_null());
        assert!((ptr as usize).is_multiple_of(BW_ALIGN));
        unsafe { bw_free(ptr, 1024) };

        assert!(bw_alloc(0).is_null());
        // Freeing null or a zero length must be a no-op, not a crash.
        unsafe { bw_free(std::ptr::null_mut(), 16) };
    }

    #[test]
    fn abi_version_is_exported() {
        assert_eq!(bw_abi_version(), BW_ABI_VERSION);
    }

    #[test]
    fn rift_api_is_additive_and_writes_all_outputs() {
        assert_eq!(bw_rift_api_version(), BW_RIFT_API_VERSION);
        assert_eq!(std::mem::size_of::<BwRiftOptionsV1>(), 48);
        assert_eq!(std::mem::size_of::<BwRiftStatsV1>(), 64);

        let (rows, gates) = (16, 64);
        let total = rows * gates;
        let velocity = vec![5.0f32; total];
        let azimuth: Vec<f32> = (0..rows).map(|row| row as f32 * 22.5).collect();
        let nyquist = vec![NYQUIST; rows];
        let mut out = vec![f32::NAN; total];
        let mut folds = vec![i8::MIN; total];
        let mut confidence = vec![u8::MAX; total];
        let mut reasons = vec![u16::MAX; total];
        let mut stats = BwRiftStatsV1::default();
        let code = unsafe {
            bw_dealias_rift_v1(
                velocity.as_ptr(),
                azimuth.as_ptr(),
                nyquist.as_ptr(),
                rows,
                gates,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                out.as_mut_ptr(),
                folds.as_mut_ptr(),
                confidence.as_mut_ptr(),
                reasons.as_mut_ptr(),
                &mut stats,
            )
        };
        assert_eq!(code, BW_OK);
        assert!(out.iter().all(|&value| value == 5.0));
        assert!(folds.iter().all(|&fold| fold == 0));
        assert!(confidence.iter().all(|&value| value == 0));
        assert!(reasons.iter().all(|&reason| reason == 0));
        assert_eq!(stats.struct_size, 64);
        assert_eq!(stats.gates_total, total as u32);
        assert_eq!(stats.gates_refined, 0);
    }

    #[test]
    fn rift_rejects_bad_version_kind_and_alias_without_writing() {
        let (rows, gates) = (8, 8);
        let total = rows * gates;
        let velocity = vec![5.0f32; total];
        let azimuth: Vec<f32> = (0..rows).map(|row| row as f32 * 45.0).collect();
        let nyquist = vec![NYQUIST; rows];
        let reference = vec![5.0f32; total];
        let mut out = vec![1234.0f32; total];
        let bad_options = BwRiftOptionsV1 {
            struct_size: 4,
            ..BwRiftOptionsV1::default()
        };
        let code = unsafe {
            bw_dealias_rift_v1(
                velocity.as_ptr(),
                azimuth.as_ptr(),
                nyquist.as_ptr(),
                rows,
                gates,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                &bad_options,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, BW_ERR_VERSION);
        assert!(out.iter().all(|&value| value == 1234.0));

        let bad_kind = [99u8];
        let code = unsafe {
            bw_dealias_rift_v1(
                velocity.as_ptr(),
                azimuth.as_ptr(),
                nyquist.as_ptr(),
                rows,
                gates,
                reference.as_ptr(),
                std::ptr::null(),
                bad_kind.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                out.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, BW_ERR_REFERENCE);
        assert!(out.iter().all(|&value| value == 1234.0));

        let code = unsafe {
            bw_dealias_rift_v1(
                velocity.as_ptr(),
                azimuth.as_ptr(),
                nyquist.as_ptr(),
                rows,
                gates,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                velocity.as_ptr() as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, BW_ERR_ALIAS);
    }
}
