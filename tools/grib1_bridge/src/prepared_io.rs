//! Bounded parallel prepared-array writes. Python supplies exact NPY header
//! metadata; Rust streams payloads and content hashes into private files.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use sha2::{Digest, Sha256};

use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

const BUFFER_BYTES: usize = 128 * 1024;

#[repr(C)]
pub struct PreparedArray {
    pub path: *const u8,
    pub path_length: usize,
    pub header: *const u8,
    pub header_length: usize,
    pub data: *const u8,
    pub data_length: usize,
    pub hash_prefix: *const u8,
    pub hash_prefix_length: usize,
}

#[repr(C)]
pub struct PreparedHashArray {
    pub data: *const u8,
    pub data_length: usize,
    pub hash_prefix: *const u8,
    pub hash_prefix_length: usize,
}

/// Hash independent immutable prepared payloads without materializing a
/// second copy. A caller may supply read-only mapped NPY payloads.
///
/// # Safety
/// Every payload and prefix stays readable and unchanged for the entire
/// call. The result buffer holds `count * 32` bytes and overlaps no input.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_hash_prepared_arrays(
    jobs: *const PreparedHashArray, count: usize, workers: usize,
    digests: *mut u8,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if jobs.is_null() || digests.is_null() { return ERR_NULL; }
        if workers == 0 || count > usize::MAX / 32 { return ERR_DIMENSION; }
        for index in 0..count {
            let job = &*jobs.add(index);
            if job.data.is_null() || job.hash_prefix.is_null() { return ERR_NULL; }
        }
        let jobs = jobs as usize;
        let digests = digests as usize;
        crate::parallel::run_ranges(count, workers, |start, stop| {
            for index in start..stop {
                let job = unsafe { &*(jobs as *const PreparedHashArray).add(index) };
                let mut hash = Sha256::new();
                hash.update(unsafe { std::slice::from_raw_parts(
                    job.hash_prefix, job.hash_prefix_length) });
                let data = unsafe { std::slice::from_raw_parts(job.data, job.data_length) };
                for chunk in data.chunks(BUFFER_BYTES) { hash.update(chunk); }
                unsafe { std::ptr::copy_nonoverlapping(
                    hash.finalize().as_ptr(), (digests as *mut u8).add(index * 32), 32) };
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

unsafe fn write_array(job: &PreparedArray, digest: *mut u8, created: *mut u8) -> std::io::Result<()> {
    let path = std::str::from_utf8(std::slice::from_raw_parts(job.path, job.path_length))
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "prepared path is not UTF-8"))?;
    // Never truncate a file which this invocation did not create.
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    *created = 1;
    let mut stream = BufWriter::with_capacity(BUFFER_BYTES, file);
    stream.write_all(std::slice::from_raw_parts(job.header, job.header_length))?;
    let mut hash = Sha256::new();
    hash.update(std::slice::from_raw_parts(job.hash_prefix, job.hash_prefix_length));
    let data = std::slice::from_raw_parts(job.data, job.data_length);
    for chunk in data.chunks(BUFFER_BYTES) {
        stream.write_all(chunk)?;
        hash.update(chunk);
    }
    stream.flush()?;
    std::ptr::copy_nonoverlapping(hash.finalize().as_ptr(), digest, 32);
    Ok(())
}

/// Write independent arrays to their private paths. Results occupy input
/// slots; the caller publishes completed files in that same order.
/// `created` identifies exactly which private files this call owns, even
/// after an I/O error, so cleanup never removes a pre-existing file.
///
/// # Safety
/// Every input buffer stays readable for the entire call. Result buffers
/// hold `count * 32` digest bytes, `count` codes and `count` ownership bytes.
/// Result buffers are disjoint from each other and all inputs.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_write_prepared_arrays(
    jobs: *const PreparedArray, count: usize, workers: usize,
    digests: *mut u8, codes: *mut i32, created: *mut u8,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if jobs.is_null() || digests.is_null() || codes.is_null() || created.is_null() {
            return ERR_NULL;
        }
        if workers == 0 || count > usize::MAX / 32 { return ERR_DIMENSION; }
        // Initialize before starting any task, including the error path.
        std::ptr::write_bytes(created, 0, count);
        std::ptr::write_bytes(codes, 0, count);
        for index in 0..count {
            let job = &*jobs.add(index);
            if job.path.is_null() || job.header.is_null() || job.data.is_null()
                || job.hash_prefix.is_null() { return ERR_NULL; }
        }
        let jobs = jobs as usize;
        let digests = digests as usize;
        let codes = codes as usize;
        let created = created as usize;
        crate::parallel::run_ranges(count, workers, |start, stop| {
            for index in start..stop {
                // Each job owns one file and one result slot.
                let result = unsafe { write_array(
                    &*(jobs as *const PreparedArray).add(index),
                    (digests as *mut u8).add(index * 32),
                    (created as *mut u8).add(index)) };
                unsafe { *(codes as *mut i32).add(index) = match result {
                    Ok(()) => 0,
                    Err(error) => error.raw_os_error().unwrap_or(-1),
                }; }
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_hash_abi_matches_sha256_known_answer_at_every_width() {
        let data = b"abc";
        let prefix = b"";
        let jobs: Vec<_> = (0..3).map(|_| PreparedHashArray {
            data: data.as_ptr(), data_length: data.len(),
            hash_prefix: prefix.as_ptr(), hash_prefix_length: 0,
        }).collect();
        for workers in [1, 8, 32] {
            let mut result = [0u8; 96];
            assert_eq!(unsafe { gpuwm_hash_prepared_arrays(
                jobs.as_ptr(), jobs.len(), workers, result.as_mut_ptr()) }, OK);
            for digest in result.chunks(32) {
                let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
                assert_eq!(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
            }
        }
    }

    #[test]
    fn mapped_hash_abi_refuses_zero_workers_before_reading_inputs() {
        let data = b"abc";
        let job = PreparedHashArray {
            data: data.as_ptr(), data_length: data.len(),
            hash_prefix: data.as_ptr(), hash_prefix_length: 0,
        };
        let mut result = [0u8; 32];
        assert_eq!(unsafe { gpuwm_hash_prepared_arrays(
            &job, 1, 0, result.as_mut_ptr()) }, ERR_DIMENSION);
    }
}
