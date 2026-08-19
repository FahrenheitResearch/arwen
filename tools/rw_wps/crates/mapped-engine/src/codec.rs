//! The acquisition codec: a compressed agency object's plainly decoded twin.
//!
//! Port of `gpuwm.ingest.codec`.  Some agencies publish every GRIB object
//! wrapped in a byte-level compression (DWD open data wraps each field file
//! in bzip2), so the object a caller stages and HASHES is the compressed
//! one while the object a parser reads is not.  Identification is by MAGIC
//! BYTES, never by filename, because a filename is caller-controlled data
//! and the refusal must describe the bytes that were actually supplied.
//!
//! Membership is deliberately closed: a codec with no staged consumer is a
//! capability without a door, so the set grows only beside a source that
//! exercises it.

use std::io::Read;

use crate::refusal::{decode_failed, Result};

/// Magic prefix -> codec name.  The engine's entire acquisition codec set.
const MAGIC_CODECS: [(&[u8], &str); 1] = [(b"BZh", "bz2")];

/// The registered codec wrapping these bytes, or `None`.
pub fn object_codec(bytes: &[u8]) -> Option<&'static str> {
    MAGIC_CODECS
        .iter()
        .find(|(magic, _name)| bytes.starts_with(magic))
        .map(|(_magic, name)| *name)
}

/// The plainly decoded payload of one acquisition object.
///
/// An uncompressed object is returned unchanged.  Provenance — hashes,
/// manifests, record references — stays bound to the SUPPLIED path,
/// because the supplied bytes are what the acquisition delivered.
pub fn decoded_payload(bytes: Vec<u8>, label: &str) -> Result<Vec<u8>> {
    match object_codec(&bytes) {
        None => Ok(bytes),
        Some("bz2") => {
            let mut plain = Vec::new();
            bzip2::read::MultiBzDecoder::new(&bytes[..])
                .read_to_end(&mut plain)
                .map_err(|error| {
                    decode_failed(format!(
                        "acquisition object {label} carries the bz2 magic but does \
                         not decompress: {error}"
                    ))
                })?;
            Ok(plain)
        }
        Some(other) => Err(decode_failed(format!(
            "acquisition object {label} carries the {other} magic, which this \
             build has no decoder for"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn an_uncompressed_object_passes_through_byte_for_byte() {
        let bytes = b"GRIB\x00\x00\x00\x02".to_vec();
        assert_eq!(decoded_payload(bytes.clone(), "sample").unwrap(), bytes);
    }

    #[test]
    fn a_bzip2_wrapped_object_decompresses_to_its_payload() {
        let payload = b"GRIB2 payload bytes, repeated. ".repeat(64);
        let mut encoder =
            bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        encoder.write_all(&payload).unwrap();
        let wrapped = encoder.finish().unwrap();
        assert_eq!(object_codec(&wrapped), Some("bz2"));
        assert_eq!(decoded_payload(wrapped, "sample").unwrap(), payload);
    }

    #[test]
    fn a_truncated_bzip2_object_refuses_by_naming_the_codec() {
        let refusal = decoded_payload(b"BZh9truncated".to_vec(), "sample").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::DECODE_FAILED);
        assert!(refusal.message.contains("bz2 magic"));
    }

    #[test]
    fn identification_is_by_magic_not_by_filename() {
        // A file NAMED .bz2 whose bytes are plain GRIB passes through; the
        // magic is the authority.
        let bytes = b"GRIB plain".to_vec();
        assert_eq!(object_codec(&bytes), None);
        assert_eq!(decoded_payload(bytes.clone(), "thing.grib2.bz2").unwrap(), bytes);
    }
}
