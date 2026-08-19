//! The C ABI seam gpuwm's `WrfoutWriter` calls through ctypes.
//!
//! ## Why ctypes and not pyo3
//!
//! Every Rust library gpuwm already drives from Python is a cdylib behind
//! ctypes: `gpuwm_preprocess_cpu` (`gpuwm/ingest/cpu_backend.py`),
//! `region_global_dealias` (`gpuwm/obs/dealias_region.py`,
//! `gpuwm/obs/coarse_cost.py`). One loading discipline, one staging
//! path in `tools/stage_wheel_bridges.py`, one ABI-marker rule in
//! `gpuwm/bridges.py`, and no build-time bind to a CPython version --
//! a pyo3 module would need a maturin build per interpreter and would be
//! the only one of its kind in the tree.
//!
//! ## Shape
//!
//! The calls mirror the netCDF C API deliberately, because the caller
//! being ported (`gpuwm/io/wrfout.py`) is written against netCDF4-python:
//! define dimensions, define variables, attach attributes, create, write,
//! close. Adding a variable to the product tape stays a row in
//! `gpuwm/io/wrf_output_schema.py` driving `def_var` -- it is not a new
//! code path, on either side of the seam.
//!
//! Data crosses as raw bytes in the CALLER'S NATIVE endianness with the
//! external type declared alongside, which is exactly what
//! `numpy.ndarray.tobytes()` on a C-contiguous array hands over. The
//! big-endian conversion happens here, once.
//!
//! ## Contract marker
//!
//! `gpuwm_ncwrite_scan_nonfinite` is the ABI marker for this library (see
//! `BRIDGE_ABI_MARKERS` in `gpuwm/bridges.py`). It names the newest
//! capability a caller's DEFAULT path depends on, which is the rule the
//! marker follows: `gpuwm_ncwrite_write_record` held the marker while the
//! record dimension was that capability, and the read-back sweep is now
//! it. A build predating the sweep exports every other symbol, loads
//! cleanly, answers the version probe -- and then cannot verify a
//! `wrfinput`, whose export has always ended by checking every float
//! variable for a non-finite value before publishing.

use std::cell::RefCell;
use std::path::PathBuf;

use crate::error::NcWriteError;
use crate::schema::Schema;
use crate::types::{AttrValue, NcFormat, NcType, VarData};
use crate::writer::NcWriter;

/// The seam's ABI version. Bump it when a signature changes shape, never
/// for a rebuild.
pub const NCWRITE_ABI_VERSION: u32 = 1;

/// The stamp for C callers, NUL-terminated -- and the exported reference
/// that keeps the stamp bytes present in the cdylib the release ships,
/// where no `main` exists to hold them.
///
/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>` is the source revision this
/// cdylib was built from, embedded so the release cut can prove a staged
/// binary matches the commit being released by reading bytes alone
/// (`tools/build_bridge_bundle.py pin --source-rev`), never by executing
/// it.  `build.rs` injects the value; `unknown` marks a build the cut
/// must refuse (outside git, or a dirty tree).
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_ncwrite_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

const OK: i32 = 0;
const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// The last read-back sweep's offending variable names, newline
    /// separated. Held beside the error slot and fetched the same way,
    /// so the seam has one answer-retrieval idiom rather than two.
    static LAST_SCAN: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn fail(err: NcWriteError) -> i32 {
    set_error(err.to_string())
}

/// # Safety
/// `ptr` must point to `len` readable bytes, or be null when `len` is 0.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr`/`len` must describe valid UTF-8.
unsafe fn utf8<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    std::str::from_utf8(unsafe { bytes(ptr, len) }?).ok()
}

/// The seam's ABI version, for the loader's static handshake.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_ncwrite_abi_version() -> u32 {
    NCWRITE_ABI_VERSION
}

/// Copy the last error message for this thread into `buf`; returns the
/// message's full byte length (which may exceed `cap`, in which case the
/// copy is truncated).
///
/// # Safety
/// `buf` must be writable for `cap` bytes, or null when `cap` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_last_error(buf: *mut u8, cap: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let source = message.as_bytes();
        if !buf.is_null() && cap > 0 {
            let n = source.len().min(cap);
            unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), buf, n) };
        }
        source.len()
    })
}

/// Start a schema. `format`: 1 = CDF-1, 2 = CDF-2, 5 = CDF-5.
/// Returns null on a bad format code.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_ncwrite_schema_new(format: u32) -> *mut Schema {
    clear_error();
    let format = match format {
        1 => NcFormat::Classic,
        2 => NcFormat::Offset64,
        5 => NcFormat::Cdf5,
        other => {
            set_error(format!(
                "unknown classic format code {other}; expected 1 (CDF-1), \
                 2 (CDF-2) or 5 (CDF-5)"
            ));
            return std::ptr::null_mut();
        }
    };
    Box::into_raw(Box::new(Schema::new(format)))
}

/// Free a schema that was never handed to `gpuwm_ncwrite_create`.
///
/// # Safety
/// `schema` must come from `gpuwm_ncwrite_schema_new` and not have been
/// freed or consumed already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_schema_free(schema: *mut Schema) {
    if !schema.is_null() {
        drop(unsafe { Box::from_raw(schema) });
    }
}

/// Define a dimension. Returns the dimid, or -1 on error.
///
/// # Safety
/// `schema` must be a live schema pointer; `name`/`name_len` valid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_def_dim(
    schema: *mut Schema,
    name: *const u8,
    name_len: usize,
    len: u64,
    unlimited: i32,
) -> i64 {
    clear_error();
    let Some(schema) = (unsafe { schema.as_mut() }) else {
        return set_error("def_dim called with a null schema") as i64;
    };
    let Some(name) = (unsafe { utf8(name, name_len) }) else {
        return set_error("def_dim name is not valid UTF-8") as i64;
    };
    let len = match usize::try_from(len) {
        Ok(len) => len,
        Err(_) => return set_error("dimension length exceeds platform usize") as i64,
    };
    match schema.def_dim(name, len, unlimited != 0) {
        Ok(dimid) => dimid as i64,
        Err(err) => fail(err) as i64,
    }
}

/// Define a variable. `dimids` points at `ndims` dimids. Returns the
/// varid, or -1 on error.
///
/// # Safety
/// `schema` live; `name` valid UTF-8; `dimids` readable for `ndims` u64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_def_var(
    schema: *mut Schema,
    name: *const u8,
    name_len: usize,
    nc_type: u32,
    ndims: usize,
    dimids: *const u64,
) -> i64 {
    clear_error();
    let Some(schema) = (unsafe { schema.as_mut() }) else {
        return set_error("def_var called with a null schema") as i64;
    };
    let Some(name) = (unsafe { utf8(name, name_len) }) else {
        return set_error("def_var name is not valid UTF-8") as i64;
    };
    let Some(ty) = NcType::from_code(nc_type) else {
        return set_error(format!("unknown nc_type code {nc_type}")) as i64;
    };
    let ids: Vec<usize> = if ndims == 0 {
        Vec::new()
    } else {
        if dimids.is_null() {
            return set_error("def_var dimids is null but ndims > 0") as i64;
        }
        let raw = unsafe { std::slice::from_raw_parts(dimids, ndims) };
        let mut ids = Vec::with_capacity(ndims);
        for &value in raw {
            match usize::try_from(value) {
                Ok(id) => ids.push(id),
                Err(_) => return set_error("dimid exceeds platform usize") as i64,
            }
        }
        ids
    };
    match schema.def_var(name, ty, &ids) {
        Ok(varid) => varid as i64,
        Err(err) => fail(err) as i64,
    }
}

/// Attach an attribute. `varid < 0` means the global attribute list.
/// `nelems` counts BYTES for `NC_CHAR` and elements otherwise; `data` is
/// native-endian.
///
/// # Safety
/// `schema` live; `name` valid UTF-8; `data` readable for
/// `nelems * sizeof(nc_type)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_put_att(
    schema: *mut Schema,
    varid: i64,
    name: *const u8,
    name_len: usize,
    nc_type: u32,
    nelems: usize,
    data: *const u8,
) -> i32 {
    clear_error();
    let Some(schema) = (unsafe { schema.as_mut() }) else {
        return set_error("put_att called with a null schema");
    };
    let Some(name) = (unsafe { utf8(name, name_len) }) else {
        return set_error("put_att name is not valid UTF-8");
    };
    let Some(ty) = NcType::from_code(nc_type) else {
        return set_error(format!("unknown nc_type code {nc_type}"));
    };
    let byte_len = match nelems.checked_mul(ty.size()) {
        Some(len) => len,
        None => return set_error("attribute byte length overflows platform usize"),
    };
    let Some(raw) = (unsafe { bytes(data, byte_len) }) else {
        return set_error("put_att data is null but nelems > 0");
    };
    let value = match decode_attr(ty, raw, nelems) {
        Ok(value) => value,
        Err(message) => return set_error(message),
    };
    let result = if varid < 0 {
        schema.put_global_attr(name, value)
    } else {
        schema.put_var_attr(varid as usize, name, value)
    };
    match result {
        Ok(()) => OK,
        Err(err) => fail(err),
    }
}

fn decode_attr(ty: NcType, raw: &[u8], nelems: usize) -> Result<AttrValue, String> {
    macro_rules! take {
        ($t:ty, $variant:path, $width:expr) => {{
            let mut out = Vec::with_capacity(nelems);
            for index in 0..nelems {
                let start = index * $width;
                let mut word = [0u8; $width];
                word.copy_from_slice(&raw[start..start + $width]);
                out.push(<$t>::from_ne_bytes(word));
            }
            $variant(out)
        }};
    }
    Ok(match ty {
        NcType::Char => AttrValue::Text(
            std::str::from_utf8(raw)
                .map_err(|err| format!("NC_CHAR attribute is not valid UTF-8: {err}"))?
                .to_string(),
        ),
        NcType::Byte => AttrValue::Bytes(raw.iter().map(|&b| b as i8).collect()),
        NcType::UByte => AttrValue::UBytes(raw.to_vec()),
        NcType::Short => take!(i16, AttrValue::Shorts, 2),
        NcType::UShort => take!(u16, AttrValue::UShorts, 2),
        NcType::Int => take!(i32, AttrValue::Ints, 4),
        NcType::UInt => take!(u32, AttrValue::UInts, 4),
        NcType::Float => take!(f32, AttrValue::Floats, 4),
        NcType::Double => take!(f64, AttrValue::Doubles, 8),
        NcType::Int64 => take!(i64, AttrValue::Int64s, 8),
        NcType::UInt64 => take!(u64, AttrValue::UInt64s, 8),
    })
}

/// Create the file. CONSUMES the schema pointer (do not free it after a
/// successful call). Returns null on error.
///
/// # Safety
/// `schema` live and not previously consumed; `path` valid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_create(
    schema: *mut Schema,
    path: *const u8,
    path_len: usize,
) -> *mut NcWriter {
    clear_error();
    if schema.is_null() {
        set_error("create called with a null schema");
        return std::ptr::null_mut();
    }
    let schema = *unsafe { Box::from_raw(schema) };
    let Some(path) = (unsafe { utf8(path, path_len) }) else {
        set_error("create path is not valid UTF-8");
        return std::ptr::null_mut();
    };
    match NcWriter::create(PathBuf::from(path), schema) {
        Ok(writer) => Box::into_raw(Box::new(writer)),
        Err(err) => {
            fail(err);
            std::ptr::null_mut()
        }
    }
}

/// Write a whole fixed variable. `data` is native-endian, `nbytes` long.
///
/// # Safety
/// `writer` live; `data` readable for `nbytes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_write_var(
    writer: *mut NcWriter,
    varid: u64,
    nc_type: u32,
    data: *const u8,
    nbytes: usize,
) -> i32 {
    clear_error();
    let Some(writer) = (unsafe { writer.as_mut() }) else {
        return set_error("write_var called with a null writer");
    };
    let Some(raw) = (unsafe { bytes(data, nbytes) }) else {
        return set_error("write_var data is null but nbytes > 0");
    };
    with_payload(nc_type, raw, |payload| writer.write_var(varid as usize, payload))
}

/// Write one record slab of a record variable.
///
/// # Safety
/// `writer` live; `data` readable for `nbytes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_write_record(
    writer: *mut NcWriter,
    recno: u64,
    varid: u64,
    nc_type: u32,
    data: *const u8,
    nbytes: usize,
) -> i32 {
    clear_error();
    let Some(writer) = (unsafe { writer.as_mut() }) else {
        return set_error("write_record called with a null writer");
    };
    let Some(raw) = (unsafe { bytes(data, nbytes) }) else {
        return set_error("write_record data is null but nbytes > 0");
    };
    with_payload(nc_type, raw, |payload| {
        writer.write_record(recno, varid as usize, payload)
    })
}

/// Reinterpret a native-endian byte buffer as the declared external type
/// and hand it to `body`.
///
/// The alignment question is real: `numpy.tobytes()` gives no alignment
/// guarantee, so the bytes are copied into a properly aligned `Vec`
/// rather than transmuted in place. One copy per slab, bounded by the
/// slab, is the price of not having undefined behaviour on the hot path.
fn with_payload<F>(nc_type: u32, raw: &[u8], body: F) -> i32
where
    F: FnOnce(VarData<'_>) -> crate::error::Result<()>,
{
    let Some(ty) = NcType::from_code(nc_type) else {
        return set_error(format!("unknown nc_type code {nc_type}"));
    };
    let width = ty.size();
    if raw.len() % width != 0 {
        return set_error(format!(
            "{} payload of {} byte(s) is not a whole number of {}-byte elements",
            ty.name(),
            raw.len(),
            width
        ));
    }
    let count = raw.len() / width;

    macro_rules! decode {
        ($t:ty, $variant:path, $width:expr) => {{
            let mut values: Vec<$t> = Vec::with_capacity(count);
            for index in 0..count {
                let start = index * $width;
                let mut word = [0u8; $width];
                word.copy_from_slice(&raw[start..start + $width]);
                values.push(<$t>::from_ne_bytes(word));
            }
            match body($variant(&values)) {
                Ok(()) => OK,
                Err(err) => fail(err),
            }
        }};
    }

    match ty {
        NcType::Char => match body(VarData::Char(raw)) {
            Ok(()) => OK,
            Err(err) => fail(err),
        },
        NcType::UByte => match body(VarData::U8(raw)) {
            Ok(()) => OK,
            Err(err) => fail(err),
        },
        NcType::Byte => {
            let values: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
            match body(VarData::I8(&values)) {
                Ok(()) => OK,
                Err(err) => fail(err),
            }
        }
        NcType::Short => decode!(i16, VarData::I16, 2),
        NcType::UShort => decode!(u16, VarData::U16, 2),
        NcType::Int => decode!(i32, VarData::I32, 4),
        NcType::UInt => decode!(u32, VarData::U32, 4),
        NcType::Float => decode!(f32, VarData::F32, 4),
        NcType::Double => decode!(f64, VarData::F64, 8),
        NcType::Int64 => decode!(i64, VarData::I64, 8),
        NcType::UInt64 => decode!(u64, VarData::U64, 8),
    }
}

/// Patch `numrecs`, flush, fsync, and free the writer. CONSUMES the
/// pointer whether it succeeds or fails.
///
/// # Safety
/// `writer` live and not previously finished or aborted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_finish(writer: *mut NcWriter) -> i32 {
    clear_error();
    if writer.is_null() {
        return set_error("finish called with a null writer");
    }
    let writer = *unsafe { Box::from_raw(writer) };
    match writer.finish() {
        Ok(()) => OK,
        Err(err) => fail(err),
    }
}

/// Free a writer WITHOUT finishing it. The file on disk stays incomplete
/// (`numrecs` 0); the caller owns deleting it.
///
/// # Safety
/// `writer` live and not previously finished or aborted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_abort(writer: *mut NcWriter) {
    if !writer.is_null() {
        drop(unsafe { Box::from_raw(writer) });
    }
}

/// Read a finished classic file back and record which float variables
/// hold a non-finite value. Returns 0 on a completed sweep (clean OR
/// not -- the names come back through `gpuwm_ncwrite_last_scan`), -1 when
/// the sweep could not run, with the reason in `gpuwm_ncwrite_last_error`.
///
/// "Could not run" and "found nothing" are separate answers on purpose: a
/// truncated file, an HDF5 container or an unreadable path must never
/// read back as a clean verification.
///
/// # Safety
/// `path`/`path_len` must describe `path_len` readable UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_scan_nonfinite(path: *const u8, path_len: usize) -> i32 {
    clear_error();
    LAST_SCAN.with(|slot| slot.borrow_mut().clear());
    let Some(text) = (unsafe { utf8(path, path_len) }) else {
        return set_error("scan_nonfinite path is not valid UTF-8");
    };
    match crate::scan::scan_nonfinite(PathBuf::from(text)) {
        Ok(names) => {
            LAST_SCAN.with(|slot| *slot.borrow_mut() = names.join("\n"));
            OK
        }
        Err(err) => fail(err),
    }
}

/// Copy the last sweep's report for this thread into `buf` -- the
/// offending variable names, newline separated, empty for a clean file;
/// returns the report's full byte length (which may exceed `cap`, in
/// which case the copy is truncated).
///
/// # Safety
/// `buf` must be writable for `cap` bytes, or null when `cap` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_last_scan(buf: *mut u8, cap: usize) -> usize {
    LAST_SCAN.with(|slot| {
        let report = slot.borrow();
        let source = report.as_bytes();
        if !buf.is_null() && cap > 0 {
            let n = source.len().min(cap);
            unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), buf, n) };
        }
        source.len()
    })
}

/// How many records the writer has seen, for a caller that wants to check
/// its own bookkeeping before finishing. Returns `u64::MAX` on a null
/// pointer.
///
/// # Safety
/// `writer` live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_ncwrite_num_records(writer: *const NcWriter) -> u64 {
    match unsafe { writer.as_ref() } {
        Some(writer) => writer.num_records(),
        None => u64::MAX,
    }
}
