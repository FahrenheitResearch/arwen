//! The two digests the frameset carries, byte-for-byte as Python computes
//! them.
//!
//! A digest is an identity, so the port is only useful if it is EXACT.
//! Both functions below are transcriptions of `gpuwm.mapped_source`:
//! `_sha256` over file bytes, and `_array_sha256` over an array's dtype
//! string, its shape rendered as Python's `json.dumps` renders a tuple
//! (`"[2, 3]"` — note the space after the comma, which is `json.dumps`'s
//! DEFAULT separator, not the compact one used for the grid fingerprint),
//! and the contiguous little-endian payload.  A missing space there would
//! produce a different hash for every field in every receipt.

use sha2::{Digest, Sha256};

/// `hashlib.sha256` over a byte string.
pub fn bytes_sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex(&digest.finalize())
}

/// `mapped_source._sha256`: the digest of a file's bytes.
pub fn file_sha256(path: &str) -> std::io::Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex(&digest.finalize()))
}

/// `mapped_source._array_sha256` for a C-contiguous `<f8` array.
pub fn array_sha256(shape: &[usize], values: &[f64]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"<f8");
    digest.update(b"\0");
    digest.update(python_shape_repr(shape).as_bytes());
    digest.update(b"\0");
    // Feed whole blocks rather than dispatching a digest update for every
    // eight-byte value. Keep the wire conversion explicit: native byte order
    // and floating-point arithmetic must not change an array's identity.
    let mut bytes = [0u8; 8192];
    for values in values.chunks(bytes.len() / 8) {
        for (slot, value) in bytes.chunks_exact_mut(8).zip(values) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        digest.update(&bytes[..values.len() * 8]);
    }
    hex(&digest.finalize())
}

/// `json.dumps(tuple(shape))` with CPython's default separators.
pub fn python_shape_repr(shape: &[usize]) -> String {
    let mut text = String::from("[");
    for (position, extent) in shape.iter().enumerate() {
        if position > 0 {
            text.push_str(", ");
        }
        text.push_str(&extent.to_string());
    }
    text.push(']');
    text
}

/// Finish a digest that was accumulated incrementally.
///
/// The frameset writer hashes the `frames.f64` stream as it writes it,
/// so it needs to hand in a hasher rather than a byte slice: the stream
/// is multi-gigabyte and re-reading it to digest it would double the
/// cost of every decode this seam exists to make cheaper.
pub fn hex_digest(digest: Sha256) -> String {
    hex(&digest.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_digest_matches_the_known_sha256_of_nothing() {
        assert_eq!(
            bytes_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn shape_repr_uses_pythons_default_separator_not_the_compact_one() {
        assert_eq!(python_shape_repr(&[2, 3]), "[2, 3]");
        assert_eq!(python_shape_repr(&[7]), "[7]");
    }

    #[test]
    fn array_digest_pins_the_python_transcription() {
        // Cross-checked against CPython:
        //   hashlib.sha256(b"<f8\x00[2]\x00" + np.array([1.0, 2.0]).tobytes())
        let digest = array_sha256(&[2], &[1.0, 2.0]);
        let mut expected = Sha256::new();
        expected.update(b"<f8\0[2]\0");
        expected.update(1.0f64.to_le_bytes());
        expected.update(2.0f64.to_le_bytes());
        assert_eq!(digest, hex(&expected.finalize()));
    }

    #[test]
    fn array_digest_keeps_ieee_words_across_buffer_boundaries() {
        // Independent hashlib SHA-256 answers over the dtype, JSON shape
        // and struct.pack('<Q', bits) payload. Include signed zero, signed
        // NaN payloads, infinities and a subnormal without doing arithmetic.
        let bits = [
            0u64,
            0x8000_0000_0000_0000,
            0x7ff8_0000_0000_0042,
            0xfff8_0000_0000_0073,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            1,
            0x3ff0_0000_0000_0000,
        ];
        for (length, expected) in [
            (
                0,
                "125d3178b96d475ca3c9dc168bea518931a581736dcefefcd73e6927dd046625",
            ),
            (
                1023,
                "d7de2a59ef2d99339497cc22599f031d0420c36e438e10be5c4ae4bfcde066e2",
            ),
            (
                1024,
                "a37ebdf0712bb51ffd9221ce6a7c609caf128bdf2795e82f7d95d9ebb8dc95de",
            ),
            (
                1025,
                "8ab335640b469e6e85ea29075abd618989b3ebb21bb23eed8148c324a7b17152",
            ),
        ] {
            let values: Vec<_> = (0..length)
                .map(|i| f64::from_bits(bits[i % bits.len()]))
                .collect();
            assert_eq!(array_sha256(&[length], &values), expected);
        }
    }
}
