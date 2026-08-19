//! LANE 1.  Byte-deterministic NPZ writer (`_write_deterministic_npz`).
//!
//! Same arrays in, same file bytes out: NPY 1.0 members (float64,
//! C-order, shape from the field dims) inside a ZIP with STORED
//! compression, pinned 1980-01-01 timestamps, 0o600 external attrs,
//! members in sorted name order, no data descriptors.  The corridor's
//! digest-relay contract depends on this determinism, so this writer
//! reproduces CPython `zipfile` + `numpy.lib.format` byte-for-byte on
//! the parity fixtures (the sealed artifact digest is compared, not
//! just the arrays) -- gated by `tests/lane1_goldens.rs` against a
//! committed Python-written golden file.

use std::io::Write;
use std::path::Path;

use crate::error::{Result, StaticError};
use crate::types::{Field, FieldSet};

/// CRC-32 (IEEE, reflected), the polynomial zlib/`zipfile` use.
fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *entry = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// NPY 1.0 bytes for one float64 C-order field, exactly as
/// `numpy.lib.format.write_array` emits them (sorted-dict header,
/// 64-byte alignment padding with spaces, trailing newline).
fn npy_bytes(field: &Field) -> Vec<u8> {
    let (planes, ny, nx) = field.dims();
    let shape = match field {
        Field::Plane(_) => format!("({ny}, {nx})"),
        Field::Stack(_) => format!("({planes}, {ny}, {nx})"),
    };
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}"
    );
    // MAGIC(6) + version(2) + header-len(2) + header + pad + '\n'
    // must be a multiple of 64 (numpy ARRAY_ALIGN).
    let unpadded = 10 + header.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let mut out = Vec::with_capacity(
        10 + header.len() + pad + 1 + field.data().len() * 8,
    );
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    let header_len = (header.len() + pad + 1) as u16;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad));
    out.push(b'\n');
    for value in field.data() {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The DOS timestamp `zipfile` derives from `(1980, 1, 1, 0, 0, 0)`.
const DOS_TIME: u16 = 0;
const DOS_DATE: u16 = (1 << 5) | 1; // year 0 (=1980), month 1, day 1

/// `create_system` byte: CPython `zipfile.ZipInfo` stamps 0 on Windows
/// and 3 elsewhere, and the parity target is the Python written on the
/// same platform.
#[cfg(windows)]
const CREATE_SYSTEM: u8 = 0;
#[cfg(not(windows))]
const CREATE_SYSTEM: u8 = 3;

struct Member {
    name: String,
    crc: u32,
    size: u32,
    offset: u32,
}

/// Write `fields` to `path` deterministically (atomic: temp + rename).
///
/// Matches `_write_deterministic_npz`: members in sorted name order,
/// STORED, pinned timestamp, external attr 0o600 << 16, version 2.0
/// fields exactly as CPython's `zipfile` writes them for small files
/// (version 20, flags 0, no data descriptor, classic end record).
pub fn write_deterministic_npz(path: &Path, fields: &FieldSet) -> Result<()> {
    let mut payload: Vec<u8> = Vec::new();
    let mut members: Vec<Member> = Vec::new();

    // BTreeMap iterates in sorted name order, matching sorted(fields).
    for (name, field) in &fields.fields {
        let data = npy_bytes(field);
        let member_name = format!("{name}.npy");
        if member_name.len() > u16::MAX as usize {
            return Err(StaticError::Invalid(format!(
                "NPZ member name too long: {member_name:?}"
            )));
        }
        let crc = crc32(&data);
        let size = u32::try_from(data.len()).map_err(|_| {
            StaticError::Invalid(format!(
                "NPZ member {member_name:?} exceeds 4 GiB; the corridor \
                 seal writes classic ZIP records"
            ))
        })?;
        let offset = payload.len() as u32;
        // local file header
        payload.extend_from_slice(&0x0403_4B50u32.to_le_bytes());
        payload.extend_from_slice(&20u16.to_le_bytes()); // version needed
        payload.extend_from_slice(&0u16.to_le_bytes()); // flags
        payload.extend_from_slice(&0u16.to_le_bytes()); // STORED
        payload.extend_from_slice(&DOS_TIME.to_le_bytes());
        payload.extend_from_slice(&DOS_DATE.to_le_bytes());
        payload.extend_from_slice(&crc.to_le_bytes());
        payload.extend_from_slice(&size.to_le_bytes()); // compressed
        payload.extend_from_slice(&size.to_le_bytes()); // uncompressed
        payload.extend_from_slice(&(member_name.len() as u16).to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes()); // extra len
        payload.extend_from_slice(member_name.as_bytes());
        payload.extend_from_slice(&data);
        members.push(Member { name: member_name, crc, size, offset });
    }

    let central_start = payload.len() as u32;
    for member in &members {
        payload.extend_from_slice(&0x0201_4B50u32.to_le_bytes());
        payload.push(20); // create version
        payload.push(CREATE_SYSTEM);
        payload.extend_from_slice(&20u16.to_le_bytes()); // version needed
        payload.extend_from_slice(&0u16.to_le_bytes()); // flags
        payload.extend_from_slice(&0u16.to_le_bytes()); // STORED
        payload.extend_from_slice(&DOS_TIME.to_le_bytes());
        payload.extend_from_slice(&DOS_DATE.to_le_bytes());
        payload.extend_from_slice(&member.crc.to_le_bytes());
        payload.extend_from_slice(&member.size.to_le_bytes());
        payload.extend_from_slice(&member.size.to_le_bytes());
        payload.extend_from_slice(&(member.name.len() as u16).to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes()); // extra len
        payload.extend_from_slice(&0u16.to_le_bytes()); // comment len
        payload.extend_from_slice(&0u16.to_le_bytes()); // disk number
        payload.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        payload.extend_from_slice(&(0o600u32 << 16).to_le_bytes());
        payload.extend_from_slice(&member.offset.to_le_bytes());
        payload.extend_from_slice(member.name.as_bytes());
    }
    let central_size = payload.len() as u32 - central_start;
    payload.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
    payload.extend_from_slice(&0u16.to_le_bytes()); // this disk
    payload.extend_from_slice(&0u16.to_le_bytes()); // central-dir disk
    payload.extend_from_slice(&(members.len() as u16).to_le_bytes());
    payload.extend_from_slice(&(members.len() as u16).to_le_bytes());
    payload.extend_from_slice(&central_size.to_le_bytes());
    payload.extend_from_slice(&central_start.to_le_bytes());
    payload.extend_from_slice(&0u16.to_le_bytes()); // comment len

    // atomic: temp beside the target, then rename over it.
    let parent = path.parent().ok_or_else(|| {
        StaticError::Invalid(format!(
            "NPZ path has no parent directory: {}",
            path.display()
        ))
    })?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let temporary =
        parent.join(format!(".tmp-{:08x}-{nonce:08x}.npz", std::process::id()));
    let write_result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&payload)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(StaticError::Io(error));
    }
    if path.exists() {
        // os.replace semantics: clobber an existing destination.
        let _ = std::fs::remove_file(path);
    }
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(StaticError::Io(error));
    }
    Ok(())
}
