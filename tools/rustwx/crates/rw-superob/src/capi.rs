//! The C ABI `gpuwm/obs/superob_bridge.py` loads through ctypes.
//!
//! Same discipline as the other rustwx cdylibs (obs-regrid, static-fields,
//! netcdf-writer): cdylib + ctypes, an ABI version probe, a contract
//! marker symbol (`gpuwm_superob_call`, listed in `gpuwm/bridges.py`
//! `BRIDGE_ABI_MARKERS`), and no panic ever crossing the boundary.
//!
//! One entry point carries every operation: a JSON request (`"op"` names
//! it) plus a table of raw buffers the request refers to by index.  The
//! answer is a JSON string the caller frees with `gpuwm_superob_free`:
//! `{"ok": true, ...}` or `{"ok": false, "kind": ..., "message": ...}`.
//! The operations are stateless; every output lands in a caller buffer.

use std::ffi::{CString, c_char};

use serde_json::{Value, json};

use crate::request::{Arrays, SoArray};
use crate::{Result, SuperobFailure};

/// Bump when the request or buffer contract changes shape.
pub const SUPEROB_ABI_VERSION: u32 = 1;

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_superob_abi_version() -> u32 {
    SUPEROB_ABI_VERSION
}

/// The source-revision stamp the release cut reads out of the binary.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_superob_source_rev() -> *const c_char {
    static SOURCE_REV_STAMP: &str =
        concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
    SOURCE_REV_STAMP.as_ptr().cast()
}

fn dispatch(request: &[u8], arrays: &Arrays) -> Result<Value> {
    let value: Value = serde_json::from_slice(request)
        .map_err(|error| SuperobFailure::request(format!("request is not JSON: {error}")))?;
    let op = value
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| SuperobFailure::request("request names no op"))?
        .to_string();
    let parse_error = |error: serde_json::Error| SuperobFailure::request(format!("{op}: {error}"));
    match op.as_str() {
        "window" => {
            #[derive(serde::Deserialize)]
            struct WindowRequest {
                lat: usize,
                lon: usize,
                ny: usize,
                nx: usize,
                #[serde(deserialize_with = "crate::request::de_f64")]
                site_lat_deg: f64,
                #[serde(deserialize_with = "crate::request::de_f64")]
                site_lon_deg: f64,
                #[serde(deserialize_with = "crate::request::de_f64")]
                earth_radius_m: f64,
                #[serde(deserialize_with = "crate::request::de_f64")]
                reach_m: f64,
            }
            let r: WindowRequest = serde_json::from_value(value).map_err(parse_error)?;
            let (j0, j1, i0, i1) = crate::grid::horizontal_window(
                arrays.f64s(r.lat, "lat")?,
                arrays.f64s(r.lon, "lon")?,
                r.ny,
                r.nx,
                r.site_lat_deg,
                r.site_lon_deg,
                r.earth_radius_m,
                r.reach_m,
            )?;
            Ok(json!({"ok": true, "window": [j0, j1, i0, i1]}))
        }
        "volume" => {
            let r: crate::volume::VolumeRequest = serde_json::from_value(value).map_err(parse_error)?;
            crate::volume::superob_volume(&r, arrays)
        }
        "merge" => {
            let r: crate::merge::MergeRequest = serde_json::from_value(value).map_err(parse_error)?;
            crate::merge::merge(&r, arrays)
        }
        "dealias_region" => {
            let r: crate::dealias::DealiasRequest =
                serde_json::from_value(value).map_err(parse_error)?;
            crate::dealias::dealias_region(&r, arrays)
        }
        other => Err(SuperobFailure::request(format!("unknown op {other:?}"))),
    }
}

fn answer(value: Value) -> *mut c_char {
    let text = serde_json::to_string(&value)
        .unwrap_or_else(|_| r#"{"ok":false,"kind":"internal","message":"unserializable answer"}"#.into());
    CString::new(text).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}

/// Run one operation.  Never returns null unless allocation itself failed.
///
/// # Safety
/// `request` points to `request_len` readable bytes; `arrays` to
/// `n_arrays` `SoArray` descriptors whose buffers are valid for their
/// stated lengths for the duration of the call, written ones exclusively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_superob_call(
    request: *const u8,
    request_len: usize,
    arrays: *const SoArray,
    n_arrays: usize,
) -> *mut c_char {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if request.is_null() {
            return Err(SuperobFailure::request("request is null"));
        }
        let request = unsafe { std::slice::from_raw_parts(request, request_len) };
        let table: &[SoArray] = if n_arrays == 0 || arrays.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(arrays, n_arrays) }
        };
        dispatch(request, &Arrays::new(table))
    }));
    match outcome {
        Ok(Ok(value)) => answer(value),
        Ok(Err(failure)) => answer(json!({"ok": false, "kind": failure.kind, "message": failure.message})),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|t| (*t).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".into());
            answer(json!({"ok": false, "kind": "internal",
                          "message": format!("panic in the superob seam: {detail}")}))
        }
    }
}

/// Free an answer returned by `gpuwm_superob_call`.
///
/// # Safety
/// `answer` is a pointer `gpuwm_superob_call` returned, freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_superob_free(answer: *mut c_char) {
    if !answer.is_null() {
        drop(unsafe { CString::from_raw(answer) });
    }
}
