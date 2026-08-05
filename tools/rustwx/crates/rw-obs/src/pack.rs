//! The observation pack: header, JSON metadata, one contiguous payload.
//!
//! Deliberately the same dull shape as the radar sweep pack — 8-byte magic,
//! `u32` version, `u32` metadata length, `u64` payload length, zero-padded to
//! 64 bytes, then the JSON, then the arrays — with two differences that earn
//! their keep:
//!
//! * the magic is `GPWMOBS1`, so a sweep pack handed to an observation
//!   reader is refused by its first eight bytes rather than by a schema
//!   string forty kilobytes in;
//! * the metadata type is a parameter, because three instruments describe
//!   themselves differently and a union type would make every reader carry
//!   fields for instruments it will never see.
//!
//! Arrays are little-endian and self-describing (`dtype`, `shape`, `offset`,
//! `bytes`). Gridded values are `<f8`: the scorer's contract pins float64 at
//! the seam, and a pack that stores `<f4` moves a conversion — and the chance
//! of getting it wrong — into every consumer.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{err, hex_sha256};

/// Pack file magic — eight bytes, versioned by the `u32` that follows.
pub const PACK_MAGIC: &[u8; 8] = b"GPWMOBS1";
pub const PACK_VERSION: u32 = 1;
pub const PACK_HEADER_BYTES: usize = 64;

/// The schema a gridded observation pack declares.
pub const GRID_SCHEMA: &str = "gpuwm-obs.obs-grid.v1";
/// The schema a grid-geometry pack declares.
pub const GEO_SCHEMA: &str = "gpuwm-obs.obs-geo.v1";

/// One little-endian array inside the payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayEntry {
    /// `<f8` or `<u1`. Spelled the way `numpy.dtype` reads it.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
}

impl ArrayEntry {
    /// Bytes one element of `dtype` occupies, or `None` if this build does
    /// not write that dtype.
    pub fn item_size(dtype: &str) -> Option<usize> {
        match dtype {
            "<f8" => Some(8),
            "<u1" => Some(1),
            _ => None,
        }
    }
}

/// Accumulates arrays into one contiguous payload.
#[derive(Default)]
pub struct PayloadBuilder {
    payload: Vec<u8>,
    arrays: BTreeMap<String, ArrayEntry>,
}

impl PayloadBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one `f64` array under `key`.
    pub fn push_f64(&mut self, key: &str, values: &[f64], shape: Vec<usize>) {
        let expected: usize = shape.iter().product();
        debug_assert_eq!(expected, values.len(), "declared shape must match the data");
        let offset = self.payload.len();
        self.payload.reserve(values.len() * 8);
        for value in values {
            self.payload.extend_from_slice(&value.to_le_bytes());
        }
        self.arrays.insert(
            key.to_string(),
            ArrayEntry {
                dtype: "<f8".to_string(),
                shape,
                offset,
                bytes: values.len() * 8,
            },
        );
    }

    /// Append one boolean mask under `key`, one byte per cell.
    ///
    /// `<u1` rather than a bit-packed mask: numpy reads it as `bool_` with a
    /// view and no arithmetic, and a mask is 1/8th of the values it guards,
    /// so the packing would buy nothing anyone can measure.
    pub fn push_mask(&mut self, key: &str, values: &[bool], shape: Vec<usize>) {
        let expected: usize = shape.iter().product();
        debug_assert_eq!(expected, values.len(), "declared shape must match the data");
        let offset = self.payload.len();
        self.payload.reserve(values.len());
        for value in values {
            self.payload.push(u8::from(*value));
        }
        self.arrays.insert(
            key.to_string(),
            ArrayEntry {
                dtype: "<u1".to_string(),
                shape,
                offset,
                bytes: values.len(),
            },
        );
    }

    pub fn finish(self) -> (Vec<u8>, BTreeMap<String, ArrayEntry>) {
        (self.payload, self.arrays)
    }
}

/// Serialize a pack: 64-byte header, JSON metadata, payload.
pub fn encode_pack<M: Serialize>(meta: &M, payload: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let meta_json = serde_json::to_vec(meta)?;
    let mut out = Vec::with_capacity(PACK_HEADER_BYTES + meta_json.len() + payload.len());
    out.extend_from_slice(PACK_MAGIC);
    out.extend_from_slice(&PACK_VERSION.to_le_bytes());
    out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.resize(PACK_HEADER_BYTES, 0);
    out.extend_from_slice(&meta_json);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Read a pack back, checking every self-describing field the writer
/// promised: magic, version, declared lengths against the file's own, and
/// the payload digest the metadata states.
pub fn decode_pack<M: DeserializeOwned>(bytes: &[u8]) -> Result<(M, Vec<u8>), Box<dyn Error>> {
    if bytes.len() < PACK_HEADER_BYTES {
        return Err(err(format!(
            "not an observation pack: {} bytes, the header alone is {PACK_HEADER_BYTES}",
            bytes.len()
        )));
    }
    if &bytes[..8] != PACK_MAGIC {
        return Err(err(format!(
            "not an observation pack: magic {:?}",
            String::from_utf8_lossy(&bytes[..8])
        )));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != PACK_VERSION {
        return Err(err(format!(
            "observation pack version {version}, this build reads {PACK_VERSION}"
        )));
    }
    let meta_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let payload_len = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]) as usize;
    let meta_end = PACK_HEADER_BYTES
        .checked_add(meta_len)
        .ok_or_else(|| err("observation pack metadata length overflows"))?;
    let payload_end = meta_end
        .checked_add(payload_len)
        .ok_or_else(|| err("observation pack payload length overflows"))?;
    if payload_end != bytes.len() {
        return Err(err(format!(
            "truncated observation pack: header declares {payload_end} bytes, file has {}",
            bytes.len()
        )));
    }
    let meta: M = serde_json::from_slice(&bytes[PACK_HEADER_BYTES..meta_end])?;
    Ok((meta, bytes[meta_end..payload_end].to_vec()))
}

/// Prove that every declared array lies inside the payload it indexes and
/// that its shape, dtype and byte count agree.
///
/// The header digest says the payload is the payload the writer hashed; this
/// says the index describes it. Neither implies the other: a correct digest
/// over a payload whose array table claims one extra row still reads past
/// the end in the consumer.
pub fn validate_arrays(
    arrays: &BTreeMap<String, ArrayEntry>,
    payload_len: usize,
) -> Result<(), Box<dyn Error>> {
    for (key, entry) in arrays {
        let item = ArrayEntry::item_size(&entry.dtype).ok_or_else(|| {
            err(format!(
                "pack array {key} declares dtype {:?}, which this build does not write",
                entry.dtype
            ))
        })?;
        if entry.shape.is_empty() {
            return Err(err(format!("pack array {key} declares no shape")));
        }
        let elements: usize = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, n| acc.checked_mul(*n))
            .ok_or_else(|| err(format!("pack array {key} shape overflows")))?;
        let declared = elements
            .checked_mul(item)
            .ok_or_else(|| err(format!("pack array {key} byte count overflows")))?;
        if declared != entry.bytes {
            return Err(err(format!(
                "pack array {key} declares shape {:?} of {} ({elements} elements, \
                 {declared} bytes) but states {} bytes",
                entry.shape, entry.dtype, entry.bytes
            )));
        }
        let end = entry
            .offset
            .checked_add(entry.bytes)
            .ok_or_else(|| err(format!("pack array {key} extent overflows")))?;
        if end > payload_len {
            return Err(err(format!(
                "pack array {key} spans bytes {}..{end} of a {payload_len}-byte payload",
                entry.offset
            )));
        }
    }
    Ok(())
}

/// Write a pack atomically, so a reader never opens a half-written one.
pub fn write_pack<M: Serialize>(
    path: &Path,
    meta: &M,
    payload: &[u8],
) -> Result<usize, Box<dyn Error>> {
    let bytes = encode_pack(meta, payload)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| err(format!("cannot create {}: {e}", parent.display())))?;
        }
    }
    let temporary = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&temporary, &bytes)
        .map_err(|e| err(format!("cannot write {}: {e}", temporary.display())))?;
    // Windows `rename` refuses an existing destination; a decode re-run over
    // its own output is ordinary, so clear the way first.
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|e| err(format!("cannot replace {}: {e}", path.display())))?;
    }
    std::fs::rename(&temporary, path).map_err(|e| {
        let _ = std::fs::remove_file(&temporary);
        err(format!("cannot publish {}: {e}", path.display()))
    })?;
    Ok(bytes.len())
}

/// The digest a pack's metadata states over its payload.
pub fn payload_digest(payload: &[u8]) -> String {
    hex_sha256(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TinyMeta {
        schema: String,
        content_sha256: String,
        arrays: BTreeMap<String, ArrayEntry>,
    }

    fn one_pack() -> (TinyMeta, Vec<u8>) {
        let mut builder = PayloadBuilder::new();
        builder.push_f64("values", &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        builder.push_mask("valid", &[true, false, true, true], vec![2, 2]);
        let (payload, arrays) = builder.finish();
        let meta = TinyMeta {
            schema: GRID_SCHEMA.to_string(),
            content_sha256: payload_digest(&payload),
            arrays,
        };
        (meta, payload)
    }

    #[test]
    fn a_pack_round_trips_through_bytes_with_its_arrays_intact() {
        let (meta, payload) = one_pack();
        let bytes = encode_pack(&meta, &payload).unwrap();
        assert_eq!(&bytes[..8], PACK_MAGIC);
        let (read, read_payload): (TinyMeta, Vec<u8>) = decode_pack(&bytes).unwrap();
        assert_eq!(read, meta);
        assert_eq!(read_payload, payload);
        validate_arrays(&read.arrays, read_payload.len()).unwrap();
        // The values land where the index says, little-endian.
        let entry = &read.arrays["values"];
        assert_eq!(entry.dtype, "<f8");
        assert_eq!(entry.shape, vec![2, 2]);
        let first = f64::from_le_bytes(
            read_payload[entry.offset..entry.offset + 8].try_into().unwrap(),
        );
        assert!((first - 1.0).abs() < 1e-12);
        // The mask is one byte per cell.
        let mask = &read.arrays["valid"];
        assert_eq!(mask.bytes, 4);
        assert_eq!(&read_payload[mask.offset..mask.offset + 4], &[1, 0, 1, 1]);
    }

    #[test]
    fn a_sweep_pack_is_refused_by_its_magic_not_by_its_schema() {
        let mut bytes = encode_pack(&one_pack().0, &[]).unwrap();
        bytes[..8].copy_from_slice(b"GPWMRDR1");
        let outcome: Result<(TinyMeta, Vec<u8>), _> = decode_pack(&bytes);
        let message = outcome.unwrap_err().to_string();
        assert!(message.contains("GPWMRDR1"), "{message}");
        assert!(message.contains("not an observation pack"), "{message}");
    }

    #[test]
    fn a_truncated_pack_is_refused_rather_than_read_short() {
        let (meta, payload) = one_pack();
        let bytes = encode_pack(&meta, &payload).unwrap();
        let cut = &bytes[..bytes.len() - 8];
        let outcome: Result<(TinyMeta, Vec<u8>), _> = decode_pack(cut);
        assert!(outcome.unwrap_err().to_string().contains("truncated"));
    }

    #[test]
    fn a_future_version_is_refused_by_number() {
        let (meta, payload) = one_pack();
        let mut bytes = encode_pack(&meta, &payload).unwrap();
        bytes[8..12].copy_from_slice(&(PACK_VERSION + 1).to_le_bytes());
        let outcome: Result<(TinyMeta, Vec<u8>), _> = decode_pack(&bytes);
        assert!(outcome.unwrap_err().to_string().contains("version"));
    }

    #[test]
    fn an_array_index_that_overruns_the_payload_is_caught() {
        let (mut meta, payload) = one_pack();
        meta.arrays.get_mut("values").unwrap().shape = vec![2, 3];
        meta.arrays.get_mut("values").unwrap().bytes = 48;
        let outcome = validate_arrays(&meta.arrays, payload.len());
        let message = outcome.unwrap_err().to_string();
        assert!(message.contains("values"), "{message}");
    }

    #[test]
    fn an_array_whose_shape_and_byte_count_disagree_is_caught() {
        let (mut meta, payload) = one_pack();
        meta.arrays.get_mut("values").unwrap().shape = vec![2, 3];
        let message = validate_arrays(&meta.arrays, payload.len())
            .unwrap_err()
            .to_string();
        assert!(message.contains("but states"), "{message}");
    }

    #[test]
    fn an_unknown_dtype_is_refused_rather_than_guessed_at() {
        let (mut meta, payload) = one_pack();
        meta.arrays.get_mut("values").unwrap().dtype = ">f8".to_string();
        let message = validate_arrays(&meta.arrays, payload.len())
            .unwrap_err()
            .to_string();
        assert!(message.contains("does not write"), "{message}");
    }

    #[test]
    fn write_pack_publishes_atomically_and_can_overwrite_its_own_output() {
        let dir = std::env::temp_dir().join(format!("rw-obs-pack-{}", std::process::id()));
        let path = dir.join("one.obspack");
        let (meta, payload) = one_pack();
        let written = write_pack(&path, &meta, &payload).unwrap();
        assert_eq!(written, std::fs::read(&path).unwrap().len());
        // A second decode over the same destination is ordinary work.
        write_pack(&path, &meta, &payload).unwrap();
        let (read, _): (TinyMeta, Vec<u8>) = decode_pack(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(read.schema, GRID_SCHEMA);
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "a temporary file survived the write");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
