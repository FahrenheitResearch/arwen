// SPDX-License-Identifier: Apache-2.0
// Original generic verification infrastructure, written for the NextGen fleet.
// This is not a numerical reference implementation or a native BMI adapter.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    ffi::{c_char, CStr},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Instant,
};

pub type Fallible<T> = Result<T, String>;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Seal {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub name: String,
    pub revision: String,
    pub licence: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub origin: String,
    pub sources: Vec<Source>,
    pub evidence: Vec<Seal>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    F64,
    I32,
    I64,
    U8,
}
impl Dtype {
    pub fn width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
    pub order: String,
    pub expected: Seal,
    pub actual: PathBuf,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub unit: String,
    pub case: String,
    pub numerics: String,
    pub provenance: Provenance,
    pub candidate_revision: String,
    pub command: Vec<String>,
    pub fields: Vec<Field>,
}

#[derive(Serialize)]
pub struct Difference {
    pub word_index: usize,
    pub expected_bits: String,
    pub actual_bits: String,
}
#[derive(Serialize)]
pub struct FieldResult {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub order: String,
    pub units: Option<String>,
    pub words: usize,
    pub different_words: usize,
    pub expected_sha256: String,
    pub actual_sha256: String,
    pub first_difference: Option<Difference>,
}
#[derive(Serialize)]
pub struct Verdict {
    pub schema: &'static str,
    pub unit: String,
    pub case: String,
    pub numerics: String,
    pub provenance: Provenance,
    pub candidate_revision: String,
    pub command: Vec<String>,
    pub request_sha256: String,
    pub bitwise_match: bool,
    pub compared_words: usize,
    pub different_words: usize,
    pub fields: Vec<FieldResult>,
    pub wall_ms: f64,
    pub parallelism: String,
}

fn digest(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

pub fn file_sha256(path: &Path) -> Fallible<String> {
    let mut f = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = f.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_hash(want: &str) -> Fallible<()> {
    if want.len() != 64 || !want.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("SHA-256 must contain exactly 64 hexadecimal digits".into());
    }
    Ok(())
}

pub fn sealed_bytes(seal: &Seal) -> Fallible<Vec<u8>> {
    validate_hash(&seal.sha256)?;
    let bytes = fs::read(&seal.path).map_err(|e| format!("{}: {e}", seal.path.display()))?;
    let actual = digest(&bytes);
    if !actual.eq_ignore_ascii_case(&seal.sha256) {
        return Err(format!(
            "{}: SHA-256 mismatch: {actual}",
            seal.path.display()
        ));
    }
    Ok(bytes)
}

fn bits(data: &[u8]) -> String {
    // Little-endian stored words displayed conventionally, most significant byte first.
    let mut result = String::from("0x");
    for b in data.iter().rev() {
        use std::fmt::Write;
        write!(&mut result, "{b:02x}").unwrap();
    }
    result
}

fn compare_field(field: &Field) -> Fallible<FieldResult> {
    if !matches!(field.order.as_str(), "fortran" | "c") {
        return Err(format!("{}: order must be fortran or c", field.name));
    }
    let count = field
        .shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| format!("{}: shape overflow", field.name))?;
    let width = field.dtype.width();
    let nbytes = count.checked_mul(width).ok_or("byte count overflow")?;
    let expected = sealed_bytes(&field.expected)?;
    let actual = fs::read(&field.actual).map_err(|e| format!("{}: {e}", field.actual.display()))?;
    if actual.len() != nbytes || expected.len() != nbytes {
        return Err(format!(
            "{}: shape/dtype require {nbytes} bytes, expected {}, actual {}",
            field.name,
            expected.len(),
            actual.len()
        ));
    }
    let (different, first) = expected
        .par_chunks_exact(width)
        .zip(actual.par_chunks_exact(width))
        .enumerate()
        .map(|(i, (a, b))| if a == b { (0usize, usize::MAX) } else { (1, i) })
        .reduce(|| (0, usize::MAX), |a, b| (a.0 + b.0, a.1.min(b.1)));
    Ok(FieldResult {
        name: field.name.clone(),
        dtype: field.dtype,
        shape: field.shape.clone(),
        order: field.order.clone(),
        units: field.units.clone(),
        words: count,
        different_words: different,
        expected_sha256: digest(&expected),
        actual_sha256: digest(&actual),
        first_difference: (different != 0).then(|| Difference {
            word_index: first,
            expected_bits: bits(&expected[first * width..(first + 1) * width]),
            actual_bits: bits(&actual[first * width..(first + 1) * width]),
        }),
    })
}

pub fn compare(request_bytes: &[u8]) -> Fallible<Verdict> {
    let start = Instant::now();
    let request: Request = serde_json::from_slice(request_bytes).map_err(|e| e.to_string())?;
    if request.schema != "nextgen.raw.compare/v1" {
        return Err("unsupported request schema".into());
    }
    if request.fields.is_empty() {
        return Err("no fields: request would check no output".into());
    }
    if request.unit.is_empty()
        || request.case.is_empty()
        || request.candidate_revision.is_empty()
        || request.command.is_empty()
        || request.provenance.origin.is_empty()
        || request.provenance.sources.is_empty()
        || request.provenance.evidence.is_empty()
    {
        return Err("unit, case, candidate, command and sealed provenance must be supplied".into());
    }
    if !matches!(request.numerics.as_str(), "strict" | "fast") {
        return Err("numerics must be strict or fast".into());
    }
    let mut names = HashSet::new();
    for f in &request.fields {
        if f.name.is_empty() || !names.insert(&f.name) {
            return Err("empty or duplicate field name".into());
        }
    }
    for source in &request.provenance.sources {
        if source.name.is_empty() || source.revision.is_empty() || source.licence.is_empty() {
            return Err("source name, revision and licence are required".into());
        }
    }
    request
        .provenance
        .evidence
        .par_iter()
        .map(|s| {
            validate_hash(&s.sha256)?;
            if file_sha256(&s.path)?.eq_ignore_ascii_case(&s.sha256) {
                Ok(())
            } else {
                Err(format!("{}: evidence SHA-256 mismatch", s.path.display()))
            }
        })
        .collect::<Fallible<Vec<_>>>()?;
    let fields: Vec<FieldResult> = request
        .fields
        .par_iter()
        .map(compare_field)
        .collect::<Fallible<_>>()?;
    let compared_words = fields.iter().map(|f| f.words).sum();
    let different_words = fields.iter().map(|f| f.different_words).sum();
    Ok(Verdict {
        schema: "nextgen.raw.verdict/v1",
        unit: request.unit,
        case: request.case,
        numerics: request.numerics,
        provenance: request.provenance,
        candidate_revision: request.candidate_revision,
        command: request.command,
        request_sha256: digest(request_bytes),
        bitwise_match: different_words == 0,
        compared_words,
        different_words,
        fields,
        wall_ms: start.elapsed().as_secs_f64() * 1000.,
        parallelism: format!(
            "Rust Rayon {} workers; fields and stored-word chunks; offline only",
            rayon::current_num_threads()
        ),
    })
}

// Caller owns buffers, capacity and lifetime; C ABI is setup/offline I/O only.
// It never initializes CUDA, computes model fields, or runs in a hydro step.
unsafe fn ctext<'a>(p: *const c_char) -> Fallible<&'a str> {
    if p.is_null() {
        return Err("null text pointer".into());
    }
    CStr::from_ptr(p).to_str().map_err(|e| e.to_string())
}
unsafe fn ffi(error: *mut c_char, cap: usize, action: impl FnOnce() -> Fallible<()>) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .unwrap_or_else(|_| Err("Rust staging operation panicked".into()));
    let message = result.as_ref().err().map(String::as_bytes).unwrap_or(b"");
    if !error.is_null() && cap > 0 {
        let n = message.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(message.as_ptr(), error.cast(), n);
        *error.add(n) = 0;
    }
    if result.is_ok() {
        0
    } else {
        2
    }
}

// Owned buffers are pageable host bytes, not CUDA allocations. Tracking ownership
// lets free take just a pointer, without trusting a foreign allocation's layout.
static OWNED_HOST: OnceLock<Mutex<HashMap<usize, Vec<u8>>>> = OnceLock::new();

fn own_host(mut payload: Vec<u8>) -> Fallible<*mut u8> {
    // Keep empty allocations distinct and non-null while exposing zero bytes.
    if payload.is_empty() {
        payload.try_reserve_exact(1).map_err(|e| e.to_string())?;
        payload.push(0);
    }
    let pointer = payload.as_mut_ptr();
    OWNED_HOST
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "owned host registry poisoned")?
        .insert(pointer as usize, payload);
    Ok(pointer)
}

/// Allocate zero-initialized Rust-owned host bytes; free with nextgen_raw_free.
/// # Safety
/// Output is writable for one pointer and does not already own an allocation.
/// Error has cap writable bytes. Await all CUDA copies before freeing the result.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_alloc(
    bytes: usize,
    output: *mut *mut u8,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if output.is_null() {
            return Err("null owned-buffer output".into());
        }
        *output = std::ptr::null_mut();
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(bytes)
            .map_err(|e| e.to_string())?;
        payload.resize(bytes, 0);
        *output = own_host(payload)?;
        Ok(())
    })
}

/// Load and validate raw bytes, returning a Rust-owned host pointer on success.
/// # Safety
/// Text is NUL-terminated UTF-8; output/error satisfy nextgen_raw_alloc's contract.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_load_owned(
    path: *const c_char,
    sha256: *const c_char,
    bytes: usize,
    output: *mut *mut u8,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if output.is_null() {
            return Err("null owned-buffer output".into());
        }
        *output = std::ptr::null_mut();
        let payload = sealed_bytes(&Seal {
            path: PathBuf::from(ctext(path)?),
            sha256: ctext(sha256)?.into(),
        })?;
        if payload.len() != bytes {
            return Err(format!(
                "payload is {} bytes, requested {bytes}",
                payload.len()
            ));
        }
        *output = own_host(payload)?;
        Ok(())
    })
}

/// Free a live Rust-owned host pointer; NULL is a no-op, unknown pointers fail.
/// # Safety
/// Caller has completed every read/write/CUDA copy using the allocation and
/// relinquishes it exclusively. Error has cap writable bytes. Never free a
/// pinned caller-owned buffer here; use its original allocator.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_free(pointer: *mut u8, error: *mut c_char, cap: usize) -> i32 {
    ffi(error, cap, || {
        if pointer.is_null() {
            return Ok(());
        }
        OWNED_HOST
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|_| "owned host registry poisoned")?
            .remove(&(pointer as usize))
            .ok_or("pointer is not a live Rust-owned host allocation")?;
        Ok(())
    })
}

/// Read sealed bytes to a caller-owned host buffer. Failed validation copies no bytes.
/// # Safety
/// Text is NUL-terminated UTF-8; destination and error buffers have the stated capacity.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_load(
    path: *const c_char,
    sha256: *const c_char,
    destination: *mut u8,
    bytes: usize,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if destination.is_null() {
            return Err("null destination".into());
        }
        let payload = sealed_bytes(&Seal {
            path: PathBuf::from(ctext(path)?),
            sha256: ctext(sha256)?.into(),
        })?;
        if payload.len() != bytes {
            return Err(format!(
                "payload is {} bytes, destination requires {bytes}",
                payload.len()
            ));
        }
        std::ptr::copy_nonoverlapping(payload.as_ptr(), destination, bytes);
        Ok(())
    })
}

/// Copy a byte range from a sealed packed file into a caller-owned host buffer.
/// The SHA covers the entire file. Validation failures copy no bytes. This
/// setup/export operation reads and hashes the entire file on each call.
/// # Safety
/// Text is NUL-terminated UTF-8; destination and error buffers have the stated
/// capacity. Destination is a host pointer, never a CUDA device pointer.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_load_slice(
    path: *const c_char,
    sha256: *const c_char,
    offset: usize,
    destination: *mut u8,
    bytes: usize,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if destination.is_null() {
            return Err("null destination".into());
        }
        let end = offset.checked_add(bytes).ok_or("byte range overflow")?;
        let payload = sealed_bytes(&Seal {
            path: PathBuf::from(ctext(path)?),
            sha256: ctext(sha256)?.into(),
        })?;
        let selected = payload
            .get(offset..end)
            .ok_or_else(|| format!("byte range {offset}..{end} exceeds {} bytes", payload.len()))?;
        std::ptr::copy_nonoverlapping(selected.as_ptr(), destination, bytes);
        Ok(())
    })
}

/// Write caller-owned host bytes to a new file; existing files are never overwritten.
/// # Safety
/// Source is readable for bytes; text and error buffers are valid for their stated sizes.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_store(
    path: *const c_char,
    source: *const u8,
    bytes: usize,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if source.is_null() {
            return Err("null source".into());
        }
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(ctext(path)?)
            .map_err(|e| e.to_string())?;
        f.write_all(std::slice::from_raw_parts(source, bytes))
            .map_err(|e| e.to_string())
    })
}

/// Hash an existing raw file into a caller-owned 65-byte hex/NUL buffer.
/// # Safety
/// Path is NUL-terminated UTF-8; digest has 65 writable bytes and error has cap bytes.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_sha256(
    path: *const c_char,
    digest: *mut c_char,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if digest.is_null() {
            return Err("null digest destination".into());
        }
        let hash = file_sha256(Path::new(ctext(path)?))?;
        std::ptr::copy_nonoverlapping(hash.as_ptr(), digest.cast(), 64);
        *digest.add(64) = 0;
        Ok(())
    })
}

#[repr(C)]
pub struct ReadItem {
    pub path: *const c_char,
    pub sha256: *const c_char,
    pub destination: *mut u8,
    pub bytes: usize,
}
#[repr(C)]
pub struct WriteItem {
    pub path: *const c_char,
    pub source: *const u8,
    pub bytes: usize,
}

/// Load independent sealed raw fields in parallel to nonoverlapping host buffers.
/// An error can leave other items completed; discard the whole batch on failure.
/// # Safety
/// Items/text/buffers remain valid until return; destinations are exclusively owned.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_read_many(
    items: *const ReadItem,
    count: usize,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if count == 0 {
            return Ok(());
        }
        if items.is_null() {
            return Err("null read items".into());
        }
        let jobs = std::slice::from_raw_parts(items, count)
            .iter()
            .map(|item| {
                if item.destination.is_null() {
                    return Err("null batch destination".into());
                }
                let begin = item.destination as usize;
                let end = begin
                    .checked_add(item.bytes)
                    .ok_or("destination range overflow")?;
                let seal = Seal {
                    path: PathBuf::from(ctext(item.path)?),
                    sha256: ctext(item.sha256)?.into(),
                };
                Ok((seal, begin, end))
            })
            .collect::<Fallible<Vec<_>>>()?;
        let mut ranges: Vec<_> = jobs.iter().map(|(_, begin, end)| (*begin, *end)).collect();
        ranges.sort_unstable();
        if ranges.windows(2).any(|p| p[0].1 > p[1].0) {
            return Err("overlapping batch destinations would race".into());
        }
        jobs.into_par_iter()
            .map(|(seal, begin, end)| {
                let bytes = sealed_bytes(&seal)?;
                if bytes.len() != end - begin {
                    return Err(format!("{}: destination size differs", seal.path.display()));
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), begin as *mut u8, bytes.len());
                Ok(())
            })
            .collect::<Fallible<Vec<_>>>()?;
        Ok(())
    })
}

/// Store independent host buffers in parallel to newly created files.
/// An error can leave other items completed; this operation is not transactional.
/// # Safety
/// All text and source buffers remain readable until return; item count is valid.
#[no_mangle]
pub unsafe extern "C" fn nextgen_raw_write_many(
    items: *const WriteItem,
    count: usize,
    error: *mut c_char,
    cap: usize,
) -> i32 {
    ffi(error, cap, || {
        if count == 0 {
            return Ok(());
        }
        if items.is_null() {
            return Err("null write items".into());
        }
        let jobs = std::slice::from_raw_parts(items, count)
            .iter()
            .map(|item| {
                if item.source.is_null() {
                    return Err("null batch source".into());
                }
                Ok((
                    PathBuf::from(ctext(item.path)?),
                    item.source as usize,
                    item.bytes,
                ))
            })
            .collect::<Fallible<Vec<_>>>()?;
        let mut paths = HashSet::new();
        if jobs.iter().any(|j| !paths.insert(&j.0)) {
            return Err("duplicate batch output path".into());
        }
        jobs.par_iter()
            .map(|(path, source, bytes)| {
                let mut f = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .map_err(|e| e.to_string())?;
                f.write_all(std::slice::from_raw_parts(*source as *const u8, *bytes))
                    .map_err(|e| e.to_string())
            })
            .collect::<Fallible<Vec<_>>>()?;
        Ok(())
    })
}
