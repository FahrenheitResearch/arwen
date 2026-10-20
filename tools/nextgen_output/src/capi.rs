//! One ownership-safe boundary for Rust data-path operations.
use crate::{Result, checkpoint, module_series, netcdf, series};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Request {
    RawRead {
        path: PathBuf,
    },
    CheckpointWrite {
        path: PathBuf,
        metadata: serde_json::Value,
        arrays: Vec<checkpoint::Array>,
    },
    CheckpointRead {
        path: PathBuf,
    },
    DatasetWrite {
        path: PathBuf,
        schema: netcdf::Dataset,
    },
    DatasetRead {
        path: PathBuf,
    },
    SeriesWrite {
        plan: Box<series::Plan>,
    },
    ModuleSeriesWrite {
        plan: Box<module_series::Plan>,
    },
}
pub fn run(request: Request, payload: &[u8]) -> Result<(serde_json::Value, Vec<u8>)> {
    match request {
        Request::RawRead { path } => Ok((
            serde_json::json!({"operation":"raw-read","path":path}),
            std::fs::read(path)?,
        )),
        Request::CheckpointWrite {
            path,
            metadata,
            arrays,
        } => Ok((
            checkpoint::write(&path, &metadata, &arrays, payload)?,
            vec![],
        )),
        Request::CheckpointRead { path } => checkpoint::read(&path),
        Request::DatasetWrite { path, schema } => {
            Ok((netcdf::write(&path, &schema, payload)?, vec![]))
        }
        Request::DatasetRead { path } => {
            let (schema, bytes) = netcdf::read(&path)?;
            Ok((serde_json::json!({"schema":schema}), bytes))
        }
        Request::SeriesWrite { plan } => Ok((series::write(&plan, payload)?, vec![])),
        Request::ModuleSeriesWrite { plan } => Ok((module_series::write(&plan, payload)?, vec![])),
    }
}

#[repr(C)]
pub struct Response {
    pub status: i32,
    pub json: *mut u8,
    pub json_len: usize,
    pub payload: *mut u8,
    pub payload_len: usize,
}
fn bytes(mut value: Vec<u8>) -> (*mut u8, usize) {
    value.shrink_to_fit();
    let boxed = value.into_boxed_slice();
    let len = boxed.len();
    (Box::into_raw(boxed).cast(), len)
}
/// # Safety
/// The request and snapshot pointers are readable for the declared lengths.
/// The returned object belongs to Rust; release once with rw_nextgen_output_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rw_nextgen_output_call(
    json: *const u8,
    json_len: usize,
    payload: *const u8,
    payload_len: usize,
) -> *mut Response {
    let result = std::panic::catch_unwind(|| -> Result<(serde_json::Value, Vec<u8>)> {
        if json.is_null() || (payload.is_null() && payload_len != 0) {
            return Err("null boundary pointer".into());
        }
        let request =
            serde_json::from_slice(unsafe { std::slice::from_raw_parts(json, json_len) })?;
        let data = if payload_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(payload, payload_len) }
        };
        run(request, data)
    });
    let (status, value, data) = match result {
        Ok(Ok((value, data))) => (0, value, data),
        Ok(Err(e)) => (1, serde_json::json!({"error":e.to_string()}), vec![]),
        Err(_) => (
            2,
            serde_json::json!({"error":"Rust output boundary panicked"}),
            vec![],
        ),
    };
    let (json, json_len) = bytes(serde_json::to_vec(&value).unwrap());
    let (payload, payload_len) = bytes(data);
    Box::into_raw(Box::new(Response {
        status,
        json,
        json_len,
        payload,
        payload_len,
    }))
}
/// # Safety
/// `response` is a live pointer returned by rw_nextgen_output_call, exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rw_nextgen_output_free(response: *mut Response) {
    if !response.is_null() {
        let r = unsafe { Box::from_raw(response) };
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                r.json, r.json_len,
            )));
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                r.payload,
                r.payload_len,
            )));
        }
    }
}
