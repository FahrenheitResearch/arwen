//! Tests for error handling when reading corrupted or malformed HDF5 data.
//!
//! These tests programmatically construct or manipulate byte sequences to verify
//! that the reader returns specific error types instead of panicking.

use hdf5_reader::checksum::jenkins_lookup3;
use hdf5_reader::error::Error;
use hdf5_reader::superblock::HDF5_MAGIC;
use hdf5_reader::Hdf5File;
use std::path::Path;

/// Extract the `Err` from a `Result<Hdf5File, Error>`, panicking if `Ok`.
///
/// We cannot use `.unwrap_err()` because `Hdf5File` does not implement `Debug`.
fn expect_err(result: Result<Hdf5File, Error>) -> Error {
    match result {
        Err(e) => e,
        Ok(_) => panic!("expected error, got Ok"),
    }
}

/// Build a minimal valid v2 superblock (48 bytes) with correct checksum.
///
/// Layout (offset_size = 8):
///   [0..8)   magic
///   [8]      version = 2
///   [9]      offset_size = 8
///   [10]     length_size = 8
///   [11]     consistency_flags = 0
///   [12..20) base_address = 0
///   [20..28) extension_address = 0xFF..FF (undefined)
///   [28..36) eof_address = <file_len>
///   [36..44) root_object_header_address = 48 (just past superblock)
///   [44..48) checksum (jenkins_lookup3 over bytes 0..44)
fn build_minimal_v2_superblock() -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);

    // Magic
    buf.extend_from_slice(&HDF5_MAGIC);
    // Version
    buf.push(2);
    // offset_size
    buf.push(8);
    // length_size
    buf.push(8);
    // consistency_flags
    buf.push(0);

    // base_address = 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // extension_address = undefined
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    // eof_address (we will set this to the total buffer length later)
    let eof_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder
                                                // root_object_header_address = 48 (immediately after superblock)
    buf.extend_from_slice(&48u64.to_le_bytes());

    // Patch eof_address to the final size (48 bytes for just the superblock)
    let total_len: u64 = 48;
    buf[eof_pos..eof_pos + 8].copy_from_slice(&total_len.to_le_bytes());

    // Compute and append checksum over bytes [0..44)
    let checksum = jenkins_lookup3(&buf[0..44]);
    buf.extend_from_slice(&checksum.to_le_bytes());

    assert_eq!(buf.len(), 48);
    buf
}

/// Helper to load a fixture file if it exists.
fn fixture_bytes(name: &str) -> Option<Vec<u8>> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("testdata/hdf5");
    let path = base.join(name);
    if path.exists() {
        Some(std::fs::read(&path).unwrap())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// 1. Truncated file tests
// ---------------------------------------------------------------------------

#[test]
fn truncated_before_magic_complete() {
    // Only 4 bytes of the 8-byte magic
    let data = HDF5_MAGIC[..4].to_vec();
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic, got: {err}"
    );
}

#[test]
fn truncated_at_magic_boundary() {
    // Exactly the 8 magic bytes but nothing after
    let data = HDF5_MAGIC.to_vec();
    let err = expect_err(Hdf5File::from_vec(data));
    // Should fail reading the version byte (UnexpectedEof) or similar I/O error
    assert!(
        matches!(
            err,
            Error::UnexpectedEof { .. } | Error::Io(_) | Error::InvalidMagic
        ),
        "expected UnexpectedEof or Io error after bare magic, got: {err}"
    );
}

#[test]
fn truncated_after_version_byte() {
    // Magic + version byte, but nothing else
    let mut data = HDF5_MAGIC.to_vec();
    data.push(2); // version 2
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnexpectedEof { .. } | Error::Io(_)),
        "expected UnexpectedEof or Io after version byte, got: {err}"
    );
}

#[test]
fn truncated_mid_superblock_v2() {
    let full = build_minimal_v2_superblock();
    // Cut off in the middle (after offset_size, length_size, flags, and part of base_address)
    let data = full[..20].to_vec();
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnexpectedEof { .. } | Error::Io(_)),
        "expected UnexpectedEof when truncated mid-superblock, got: {err}"
    );
}

#[test]
fn truncated_before_checksum() {
    let full = build_minimal_v2_superblock();
    // Cut off the last 4 bytes (the checksum)
    let data = full[..44].to_vec();
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnexpectedEof { .. } | Error::Io(_)),
        "expected UnexpectedEof when checksum is missing, got: {err}"
    );
}

#[test]
fn truncated_real_file_at_various_points() {
    let bytes = match fixture_bytes("scalar_dataset.h5") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: fixture scalar_dataset.h5 not found");
            return;
        }
    };

    // Truncation points that fall within the superblock should produce errors
    // during parsing. Points beyond the superblock may still parse successfully
    // (the superblock is self-contained), so we only assert errors for small
    // truncation points and verify no panics for all.
    let must_fail_points = [0, 1, 4, 7, 8, 9, 16, 32];
    for &point in &must_fail_points {
        if point >= bytes.len() {
            continue;
        }
        let truncated = bytes[..point].to_vec();
        let result = Hdf5File::from_vec(truncated);
        assert!(
            result.is_err(),
            "expected error when truncated at byte {point}, but got Ok"
        );
    }

    // For larger truncation points, just verify no panic occurs
    let no_panic_points = [48, bytes.len() / 2, bytes.len() - 1];
    for &point in &no_panic_points {
        if point >= bytes.len() {
            continue;
        }
        let truncated = bytes[..point].to_vec();
        // We do not assert is_err here: the superblock may parse fine from a
        // truncated file. We only verify it does not panic.
        let _ = Hdf5File::from_vec(truncated);
    }
}

// ---------------------------------------------------------------------------
// 2. Bad magic bytes tests
// ---------------------------------------------------------------------------

#[test]
fn all_zeros_no_magic() {
    let data = vec![0u8; 256];
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic for all-zero data, got: {err}"
    );
}

#[test]
fn flipped_first_magic_byte() {
    let mut data = build_minimal_v2_superblock();
    // Flip the first byte of the magic (\x89 -> \x88)
    data[0] ^= 0x01;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic when first magic byte is flipped, got: {err}"
    );
}

#[test]
fn flipped_middle_magic_byte() {
    let mut data = build_minimal_v2_superblock();
    // Flip byte 3 ('F' in "HDF" -> corrupted)
    data[3] ^= 0xFF;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic when middle magic byte is flipped, got: {err}"
    );
}

#[test]
fn reversed_magic_bytes() {
    let mut data = build_minimal_v2_superblock();
    // Reverse the magic bytes
    data[..8].reverse();
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic when magic bytes are reversed, got: {err}"
    );
}

#[test]
fn wrong_magic_signature() {
    // Use PNG magic instead of HDF5
    let mut data = build_minimal_v2_superblock();
    let png_magic = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    data[..8].copy_from_slice(&png_magic);
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic for PNG magic, got: {err}"
    );
}

#[test]
fn empty_file() {
    let data = Vec::new();
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic for empty file, got: {err}"
    );
}

#[test]
fn single_byte_file() {
    let data = vec![0x89]; // Just the first byte of HDF5 magic
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic for single byte, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// 3. Checksum mismatch tests
// ---------------------------------------------------------------------------

#[test]
fn corrupted_checksum_in_v2_superblock() {
    let mut data = build_minimal_v2_superblock();
    // Flip a bit in the checksum (last 4 bytes)
    data[44] ^= 0x01;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::ChecksumMismatch { .. }),
        "expected ChecksumMismatch when checksum byte is corrupted, got: {err}"
    );
}

#[test]
fn zeroed_checksum_in_v2_superblock() {
    let mut data = build_minimal_v2_superblock();
    // Zero out the checksum
    data[44..48].fill(0x00);
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::ChecksumMismatch { .. }),
        "expected ChecksumMismatch when checksum is zeroed, got: {err}"
    );
}

#[test]
fn flipped_data_byte_triggers_checksum_mismatch() {
    let mut data = build_minimal_v2_superblock();
    // Flip a bit in the consistency_flags byte (byte 11) -- the checksum
    // was computed over the original data, so this should trigger a mismatch
    data[11] ^= 0x01;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::ChecksumMismatch { .. }),
        "expected ChecksumMismatch when data byte is corrupted, got: {err}"
    );
}

#[test]
fn corrupted_base_address_triggers_checksum_mismatch() {
    let mut data = build_minimal_v2_superblock();
    // Corrupt a byte in the base_address field (byte 12)
    data[12] ^= 0x42;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::ChecksumMismatch { .. }),
        "expected ChecksumMismatch when base_address is corrupted, got: {err}"
    );
}

#[test]
fn corrupted_real_file_checksum() {
    let mut bytes = match fixture_bytes("scalar_dataset.h5") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: fixture scalar_dataset.h5 not found");
            return;
        }
    };

    // Verify it parses successfully first
    assert!(
        Hdf5File::from_bytes(&bytes).is_ok(),
        "fixture should parse without error"
    );

    // Determine superblock version to know where the checksum is
    let version = bytes[8];
    if version >= 2 {
        let offset_size = bytes[9] as usize;
        // v2/v3 checksum is at: 8(magic) + 1(ver) + 1(off_sz) + 1(len_sz) + 1(flags)
        //                       + 4 * offset_size
        let checksum_offset = 12 + 4 * offset_size;
        if checksum_offset + 4 <= bytes.len() {
            // Flip a bit in the stored checksum
            bytes[checksum_offset] ^= 0x80;
            let err = expect_err(Hdf5File::from_vec(bytes));
            assert!(
                matches!(err, Error::ChecksumMismatch { .. }),
                "expected ChecksumMismatch for corrupted real file checksum, got: {err}"
            );
        }
    } else {
        // v0/v1 has no superblock checksum, skip
        eprintln!("SKIPPED: fixture has v0/v1 superblock (no checksum to corrupt)");
    }
}

// ---------------------------------------------------------------------------
// 4. Invalid superblock version tests
// ---------------------------------------------------------------------------

#[test]
fn superblock_version_4_unsupported() {
    let mut data = build_minimal_v2_superblock();
    // Change version byte to 4 (unsupported)
    data[8] = 4;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnsupportedSuperblockVersion(4)),
        "expected UnsupportedSuperblockVersion(4), got: {err}"
    );
}

#[test]
fn superblock_version_255_unsupported() {
    let mut data = build_minimal_v2_superblock();
    data[8] = 255;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnsupportedSuperblockVersion(255)),
        "expected UnsupportedSuperblockVersion(255), got: {err}"
    );
}

#[test]
fn superblock_version_5_unsupported() {
    let mut data = build_minimal_v2_superblock();
    data[8] = 5;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnsupportedSuperblockVersion(5)),
        "expected UnsupportedSuperblockVersion(5), got: {err}"
    );
}

#[test]
fn superblock_version_128_unsupported() {
    let mut data = build_minimal_v2_superblock();
    data[8] = 128;
    let err = expect_err(Hdf5File::from_vec(data));
    assert!(
        matches!(err, Error::UnsupportedSuperblockVersion(128)),
        "expected UnsupportedSuperblockVersion(128), got: {err}"
    );
}

// ---------------------------------------------------------------------------
// 5. Combined / edge-case corruption tests
// ---------------------------------------------------------------------------

#[test]
fn random_garbage_bytes() {
    // 1024 bytes of non-zero garbage that is unlikely to contain the HDF5 magic
    let data: Vec<u8> = (0u16..1024).map(|i| ((i * 37 + 13) % 251) as u8).collect();
    let err = expect_err(Hdf5File::from_vec(data));
    // Should fail at magic detection
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic for random garbage, got: {err}"
    );
}

#[test]
fn valid_magic_then_garbage() {
    // Valid magic followed by random garbage (no valid superblock structure)
    let mut data = HDF5_MAGIC.to_vec();
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE]);
    data.extend_from_slice(&[0xFF; 128]);

    let result = Hdf5File::from_vec(data);
    // Should not panic -- any error type is acceptable as long as no panic
    assert!(
        result.is_err(),
        "expected error for magic + garbage, but got Ok"
    );
}

#[test]
fn corrupted_offset_size_zero() {
    let mut data = build_minimal_v2_superblock();
    // Set offset_size to 0 (invalid)
    data[9] = 0;
    // Recompute checksum so we get past the checksum check and hit the
    // offset size validation (if any), or another parse error
    let checksum = jenkins_lookup3(&data[0..44]);
    data[44..48].copy_from_slice(&checksum.to_le_bytes());

    let result = Hdf5File::from_vec(data);
    // Should produce an error (UnsupportedOffsetSize or UnexpectedEof or similar)
    assert!(
        result.is_err(),
        "expected error for offset_size=0, but got Ok"
    );
}

#[test]
fn corrupted_offset_size_large() {
    let mut data = build_minimal_v2_superblock();
    // Set offset_size to 16 (unsupported -- only 2, 4, 8 are valid)
    data[9] = 16;
    // Recompute checksum
    let checksum = jenkins_lookup3(&data[0..44]);
    data[44..48].copy_from_slice(&checksum.to_le_bytes());

    let result = Hdf5File::from_vec(data);
    assert!(
        result.is_err(),
        "expected error for offset_size=16, but got Ok"
    );
}

#[test]
fn valid_parse_then_corrupted_dataset_read() {
    let mut bytes = match fixture_bytes("simple_contiguous.h5") {
        Some(b) => b,
        None => {
            eprintln!("SKIPPED: fixture simple_contiguous.h5 not found");
            return;
        }
    };

    // Corrupt bytes well past the superblock (in the data region)
    let corrupt_offset = bytes.len() * 3 / 4;
    if corrupt_offset < bytes.len() {
        for i in 0..std::cmp::min(32, bytes.len() - corrupt_offset) {
            bytes[corrupt_offset + i] ^= 0xFF;
        }
    }

    // The superblock may still parse, but reading data should fail or produce wrong values.
    // We just verify no panic occurs.
    let result = Hdf5File::from_vec(bytes);
    match result {
        Ok(file) => {
            // Even if superblock parsed, reading datasets from corrupted data should
            // either error or produce data (but not panic).
            let _ = file.root_group().and_then(|g| g.dataset("data"));
        }
        Err(_) => {
            // An error is also acceptable -- the corruption may have hit metadata.
        }
    }
}

#[test]
fn from_bytes_api_also_handles_corruption() {
    // Verify that from_bytes (not just from_vec) handles corruption the same way
    let data = [0u8; 64];
    let err = expect_err(Hdf5File::from_bytes(&data));
    assert!(
        matches!(err, Error::InvalidMagic),
        "expected InvalidMagic via from_bytes, got: {err}"
    );
}
