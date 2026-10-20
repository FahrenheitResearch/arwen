//! Offline raw CUDA transfer bridge and exact native-call record packing.
//! SPDX-License-Identifier: Apache-2.0
use std::ffi::{c_char, c_int, CStr};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

unsafe fn path<'a>(path: *const c_char) -> Result<&'a std::path::Path, String> {
    use std::os::unix::ffi::OsStrExt;
    if path.is_null() {
        return Err("null path".into());
    }
    Ok(std::path::Path::new(std::ffi::OsStr::from_bytes(
        CStr::from_ptr(path).to_bytes(),
    )))
}

fn ffi_result(result: Result<(), String>, error: *mut c_char, capacity: usize) -> c_int {
    match result {
        Ok(()) => 0,
        Err(message) => {
            if !error.is_null() && capacity > 0 {
                let bytes = message.as_bytes();
                let n = bytes.len().min(capacity - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast(), n);
                    *error.add(n) = 0;
                }
            }
            -1
        }
    }
}

/// Copy unchanged file bytes directly into caller-owned pinned host memory.
/// Setup/output only; the caller validates the fixture seal before this call.
#[no_mangle]
pub unsafe extern "C" fn troute_raw_load(
    filename: *const c_char,
    destination: *mut u8,
    bytes: usize,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    let result = (|| {
        if destination.is_null() || bytes == 0 {
            return Err("empty/null destination".into());
        }
        let mut file = File::open(path(filename)?).map_err(|e| e.to_string())?;
        if file.metadata().map_err(|e| e.to_string())?.len() != bytes as u64 {
            return Err("raw byte length disagrees with fixture shape".into());
        }
        file.read_exact(std::slice::from_raw_parts_mut(destination, bytes))
            .map_err(|e| e.to_string())
    })();
    ffi_result(result, error, capacity)
}

/// Write unchanged caller-owned pinned bytes, refusing to replace an old proof.
#[no_mangle]
pub unsafe extern "C" fn troute_raw_store(
    filename: *const c_char,
    source: *const u8,
    bytes: usize,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    let result = (|| {
        if source.is_null() || bytes == 0 {
            return Err("empty/null source".into());
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path(filename)?)
            .map_err(|e| e.to_string())?;
        file.write_all(std::slice::from_raw_parts(source, bytes))
            .map_err(|e| e.to_string())
    })();
    ffi_result(result, error, capacity)
}

pub const INPUTS: [&str; 15] = [
    "dt", "qup", "quc", "qdp", "ql", "dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0", "velp",
    "depthp",
];
pub const OUTPUTS: [&str; 6] = ["qdc", "velc", "depthc", "ck", "cn", "X"];

/// Demultiplex observed calls without interpreting or rounding any float.
pub fn split_records(raw: &[u8]) -> Result<[Vec<u8>; 3], String> {
    if raw.is_empty() || raw.len() % 108 != 0 {
        return Err("capture must contain complete nonempty 27-word records".into());
    }
    let calls = raw.len() / 108;
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(calls);
    let chunk = calls.div_ceil(workers) * 108;
    let packed: Vec<[Vec<u8>; 3]> = std::thread::scope(|scope| {
        let handles: Vec<_> = raw.chunks(chunk).map(|part| scope.spawn(move || {
            let mut arrays: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());
            for record in part.chunks_exact(108) {
                if record[60..84].iter().any(|b| *b != 0) {
                    return Err("native caller did not initialize all six output words to +0 (reach.pyx contract)".to_owned());
                }
                arrays[0].extend_from_slice(&record[..60]);
                arrays[1].extend_from_slice(&record[60..84]);
                arrays[2].extend_from_slice(&record[84..108]);
            }
            Ok(arrays)
        })).collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Result<_, String>>()
    })?;
    let mut arrays: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());
    for part in packed {
        for (array, bytes) in arrays.iter_mut().zip(part) {
            array.extend(bytes);
        }
    }
    Ok(arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserve_diagnostic_bits_and_require_caller_initialization() {
        let mut bytes = vec![0_u8; 108];
        bytes[104..108].copy_from_slice(&0x7fc01234_u32.to_le_bytes());
        let split = split_records(&bytes).unwrap();
        assert_eq!(&split[2][20..24], &0x7fc01234_u32.to_le_bytes());
        bytes[80] = 1; // Incoming X differs: CUDA's zero initialization is invalid.
        assert!(split_records(&bytes)
            .unwrap_err()
            .contains("six output words"));
        assert!(split_records(&bytes[..107]).is_err());
    }
}
